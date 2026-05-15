import os, sys, argparse, json, pickle, random, time
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, jaccard_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

# ══════════════════════════════════════════════════════════════════
# KONFIGURACIJA
# ══════════════════════════════════════════════════════════════════
CFG = {
    # Putanje
    "image_dir"   : os.path.join("data", "images"),
    "mask_dir"    : os.path.join("data", "masks"),
    "clean_dir"   : os.path.join("data", "images_clean"),
    "results_dir" : "results",

    # Dataset split
    "val_split"   : 0.15,
    "test_split"  : 0.10,
    "random_seed" : 42,

    # DL trening
    "img_size"    : 256,
    "batch_size"  : 8,
    "num_epochs"  : 60,
    "lr"          : 3e-4,

    # Random Forest
    "rf_img_size"   : 128,
    "rf_max_train"  : 400,
    "rf_pixel_ratio": 0.15,
    "rf_n_estimators": 100,
    "rf_max_depth"  : 25,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(CFG["results_dir"], exist_ok=True)
random.seed(CFG["random_seed"])
np.random.seed(CFG["random_seed"])
torch.manual_seed(CFG["random_seed"])

print(f"Uređaj: {DEVICE}")


# ══════════════════════════════════════════════════════════════════
# FAZA 1 — PREDOBRADA (uklanjanje dlačica, mjehurića, normalizacija)
# ══════════════════════════════════════════════════════════════════

def remove_hair(img_bgr: np.ndarray) -> np.ndarray:
    """
    DullRazor-like algoritam:
      1. Black-hat filter → detektira tamne linearne strukture (dlake)
      2. Binarizacija → maska dlaka
      3. Telea inpainting → rekonstrukcija piksela ispod dlake
    """
    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask  = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    dil_k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask     = cv2.dilate(mask, dil_k, iterations=1)
    return cv2.inpaint(img_bgr, mask, 6, cv2.INPAINT_TELEA)


def remove_bubbles(img_bgr: np.ndarray) -> np.ndarray:
    """
    Uklanjanje mjehurića gela (svijetle okrugle strukture):
      1. Pretvorba u LAB → kanal L (luminancija)
      2. Detekcija jako svijetlih piksela (mjehurići su bijeli)
      3. Inpainting
    """
    lab      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    l_ch     = lab[:, :, 0]
    _, mask  = cv2.threshold(l_ch, 220, 255, cv2.THRESH_BINARY)
    dil_k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask     = cv2.dilate(mask, dil_k, iterations=2)
    return cv2.inpaint(img_bgr, mask, 5, cv2.INPAINT_TELEA)


def normalize_intensity(img_bgr: np.ndarray) -> np.ndarray:
    """
    Normalizacija intenziteta piksela:
      - CLAHE (Contrast Limited Adaptive Histogram Equalization)
        na L kanalu u LAB prostoru boja
      - Poboljšava kontrast između lezije i zdrave kože
        bez prekomjernog pojačavanja šuma
    """
    lab      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)


def denoise(img_bgr: np.ndarray) -> np.ndarray:
    """
    Uklanjanje šuma: FastNlMeans (čuva rubove lezije).
    """
    return cv2.fastNlMeansDenoisingColored(img_bgr, None, 6, 6, 7, 21)


def preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
    """Cijeli predobradni pipeline na jednoj slici."""
    img = denoise(img_bgr)
    img = remove_hair(img)
    img = remove_bubbles(img)
    img = normalize_intensity(img)
    return img


def run_preprocessing():
    """Obradi sve slike i spremi u images_clean/."""
    os.makedirs(CFG["clean_dir"], exist_ok=True)
    files = sorted([f for f in os.listdir(CFG["image_dir"])
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    print(f"\n{'─'*55}")
    print(f"PREDOBRADA — {len(files)} slika")
    print(f"{'─'*55}")

    for fname in tqdm(files, desc="Predobrada"):
        src = os.path.join(CFG["image_dir"], fname)
        dst = os.path.join(CFG["clean_dir"], fname)
        if os.path.exists(dst):
            continue
        img = cv2.imread(src)
        if img is None:
            continue
        cv2.imwrite(dst, preprocess_image(img))

    print(f"✓ Očišćene slike → {CFG['clean_dir']}")
    _save_preprocess_preview(files)


def _save_preprocess_preview(files, n=4):
    """Spremi vizualni preview predobrade."""
    sample = random.sample(files, min(n, len(files)))
    rows   = []
    for fname in sample:
        orig = cv2.imread(os.path.join(CFG["image_dir"], fname))
        clean = cv2.imread(os.path.join(CFG["clean_dir"], fname))
        if orig is None or clean is None:
            continue
        h, w = 200, 200
        orig_r  = cv2.resize(orig,  (w, h))
        clean_r = cv2.resize(clean, (w, h))
        rows.append(np.concatenate([orig_r, clean_r], axis=1))

    if not rows:
        return
    canvas = np.concatenate(rows, axis=0)
    # Zaglavlje
    hdr = np.zeros((28, 400, 3), dtype=np.uint8)
    hdr_pil = Image.fromarray(cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(hdr_pil)
    draw.text((5,  7), "Original",  fill=(255,255,255))
    draw.text((205, 7), "Predobrađena", fill=(255,255,255))
    hdr = cv2.cvtColor(np.array(hdr_pil), cv2.COLOR_RGB2BGR)
    out = np.concatenate([hdr, canvas], axis=0)
    path = os.path.join(CFG["results_dir"], "preprocessing_preview.png")
    cv2.imwrite(path, out)
    print(f"  Preview predobrade → {path}")


# ══════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════

def _get_paths():
    """Vrati sortirane putanje slika i maski."""
    img_dir = CFG["clean_dir"] if os.path.exists(CFG["clean_dir"]) \
              else CFG["image_dir"]
    imgs  = sorted([f for f in os.listdir(img_dir)
                    if f.lower().endswith((".jpg",".jpeg",".png"))])
    masks = sorted([f for f in os.listdir(CFG["mask_dir"])
                    if f.lower().endswith((".jpg",".jpeg",".png"))])
    assert len(imgs) == len(masks), \
        f"Broj slika ({len(imgs)}) ≠ broj maski ({len(masks)})"
    img_paths  = [os.path.join(img_dir,          f) for f in imgs]
    mask_paths = [os.path.join(CFG["mask_dir"],  f) for f in masks]
    return img_paths, mask_paths


def _split_indices(n):
    n_test  = int(n * CFG["test_split"])
    n_val   = int(n * CFG["val_split"])
    n_train = n - n_test - n_val
    idx     = list(range(n))
    random.shuffle(idx)
    return idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]


class DermDataset(Dataset):
    def __init__(self, img_paths, mask_paths, augment=False):
        self.imgs    = img_paths
        self.masks   = mask_paths
        self.augment = augment
        self.sz      = CFG["img_size"]

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i):
        img  = Image.open(self.imgs[i]).convert("RGB")
        mask = Image.open(self.masks[i]).convert("L")

        img  = TF.resize(img,  [self.sz, self.sz])
        mask = TF.resize(mask, [self.sz, self.sz],
                         interpolation=TF.InterpolationMode.NEAREST)

        if self.augment:
            if random.random() > 0.5:
                img, mask = TF.hflip(img), TF.hflip(mask)
            if random.random() > 0.5:
                img, mask = TF.vflip(img), TF.vflip(mask)
            ang  = random.uniform(-35, 35)
            img  = TF.rotate(img,  ang)
            mask = TF.rotate(mask, ang)
            if random.random() > 0.4:
                img = TF.adjust_brightness(img, random.uniform(0.6, 1.4))
            if random.random() > 0.4:
                img = TF.adjust_contrast(img,   random.uniform(0.7, 1.3))
            if random.random() > 0.4:
                img = TF.adjust_saturation(img, random.uniform(0.7, 1.3))

        img  = TF.normalize(TF.to_tensor(img),
                            [0.485,0.456,0.406], [0.229,0.224,0.225])
        mask = (TF.to_tensor(mask) > 0.5).float()
        return img, mask


def build_dl_loaders():
    img_paths, mask_paths = _get_paths()
    n = len(img_paths)
    train_idx, val_idx, test_idx = _split_indices(n)

    def subset(idx, aug):
        return DermDataset([img_paths[i]  for i in idx],
                           [mask_paths[i] for i in idx], augment=aug)

    print(f"Dataset: {len(train_idx)} train | "
          f"{len(val_idx)} val | {len(test_idx)} test")
    # num_workers=0 potrebno na macOS (MPS + Python 3.13 multiprocessing bug)
    # pin_memory=False jer MPS ne podržava pinned memory
    tr = DataLoader(subset(train_idx, True),  CFG["batch_size"],
                    shuffle=True,  num_workers=0, pin_memory=False)
    vl = DataLoader(subset(val_idx,   False), CFG["batch_size"],
                    shuffle=False, num_workers=0, pin_memory=False)
    ts = DataLoader(subset(test_idx,  False), CFG["batch_size"],
                    shuffle=False, num_workers=0, pin_memory=False)
    return tr, vl, ts, test_idx, img_paths, mask_paths


# ══════════════════════════════════════════════════════════════════
# METRIKE
# ══════════════════════════════════════════════════════════════════

def pixel_accuracy(pred, target):
    """
    Preciznost (Pixel Accuracy):
        PA = (TP + TN) / (TP + TN + FP + FN)
    Mjeri udio ispravno klasificiranih piksela.
    """
    return (pred == target).float().mean().item()


def dice_coefficient(pred, target, smooth=1e-6):
    """
    Dice koeficijent:
        Dice = 2·|X∩Y| / (|X|+|Y|)
    Mjeri preklapanje predviđene i referentne maske.
    Prikladan za medicinske primjene zbog otpornosti
    na neuravnoteženost klasa (pozadina >> lezija).
    """
    inter = (pred * target).sum().item()
    return (2*inter + smooth) / (pred.sum().item() + target.sum().item() + smooth)


def iou_score(pred, target, smooth=1e-6):
    inter = (pred * target).sum().item()
    union = pred.sum().item() + target.sum().item() - inter
    return (inter + smooth) / (union + smooth)


@torch.no_grad()
def evaluate_dl(model, loader):
    model.eval()
    acc_l, dice_l, iou_l = [], [], []
    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        pred = (torch.sigmoid(model(imgs)) > 0.5).float()
        for i in range(pred.shape[0]):
            acc_l.append(pixel_accuracy(pred[i], masks[i]))
            dice_l.append(dice_coefficient(pred[i], masks[i]))
            iou_l.append(iou_score(pred[i], masks[i]))
    return np.mean(acc_l), np.mean(dice_l), np.mean(iou_l)


# ══════════════════════════════════════════════════════════════════
# LOSS
# ══════════════════════════════════════════════════════════════════

class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, smooth=1e-6,
                 pos_weight=4.0):
        super().__init__()
        self.alpha, self.gamma, self.smooth = alpha, gamma, smooth
        self.pw = torch.tensor([pos_weight])

    def forward(self, logits, targets):
        pw  = self.pw.to(logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, targets,
              pos_weight=pw, reduction='none')
        p   = torch.sigmoid(logits)
        pt  = torch.where(targets == 1, p, 1-p)
        focal = (self.alpha * (1-pt)**self.gamma * bce).mean()
        inter = (p * targets).sum(dim=(2,3))
        dice  = 1 - (2*inter + self.smooth) / \
                (p.sum(dim=(2,3)) + targets.sum(dim=(2,3)) + self.smooth)
        return focal + dice.mean()


# ══════════════════════════════════════════════════════════════════
# ARHITEKTURA — U-NET
# ══════════════════════════════════════════════════════════════════

class DoubleConv(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(True),
            nn.Conv2d(oc, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(True))
    def forward(self, x): return self.b(x)


class UNet(nn.Module):
    """
    U-Net: enkoder–dekoder s preskočnim vezama (skip connections).
    Skip connections prenose prostorne informacije iz enkodera
    direktno u dekoder, što omogućava preciznu rekonstrukciju rubova.
    """
    def __init__(self, ic=3, oc=1, feats=[64,128,256,512]):
        super().__init__()
        self.enc  = nn.ModuleList()
        self.ups  = nn.ModuleList()
        self.dec  = nn.ModuleList()
        self.pool = nn.MaxPool2d(2,2)
        ch = ic
        for f in feats:
            self.enc.append(DoubleConv(ch, f)); ch = f
        self.bottleneck = nn.Sequential(DoubleConv(feats[-1], feats[-1]*2),
                                        nn.Dropout2d(0.3))
        for f in reversed(feats):
            self.ups.append(nn.ConvTranspose2d(f*2, f, 2, 2))
            self.dec.append(DoubleConv(f*2, f))
        self.out = nn.Conv2d(feats[0], oc, 1)

    def forward(self, x):
        skips = []
        for e in self.enc:
            x = e(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x); skips = skips[::-1]
        for i,(u,d) in enumerate(zip(self.ups, self.dec)):
            x = u(x); s = skips[i]
            if x.shape != s.shape:
                x = TF.center_crop(x, [s.shape[2], s.shape[3]])
            x = d(torch.cat([s, x], 1))
        return self.out(x)


# ══════════════════════════════════════════════════════════════════
# ARHITEKTURA — SEGNET
# ══════════════════════════════════════════════════════════════════

def cbr(ic, oc):
    return nn.Sequential(
        nn.Conv2d(ic, oc, 3, padding=1, bias=False),
        nn.BatchNorm2d(oc), nn.ReLU(True))


class SegNet(nn.Module):
    """
    SegNet: enkoder–dekoder koji umjesto skip connections
    koristi pohranjene indekse maksimalnog združivanja (max-pooling indices)
    za precizno naduzorkovanje u dekoderu.
    Memorijski efikasniji od U-Neta.
    """
    def __init__(self, ic=3, oc=1):
        super().__init__()
        self.e1 = nn.Sequential(cbr(ic,64),  cbr(64,64))
        self.e2 = nn.Sequential(cbr(64,128), cbr(128,128))
        self.e3 = nn.Sequential(cbr(128,256),cbr(256,256),cbr(256,256))
        self.e4 = nn.Sequential(cbr(256,512),cbr(512,512),cbr(512,512))
        self.e5 = nn.Sequential(cbr(512,512),cbr(512,512),cbr(512,512))
        self.pool   = nn.MaxPool2d(2,2,return_indices=True)
        self.unpool = nn.MaxUnpool2d(2,2)
        self.d5 = nn.Sequential(cbr(512,512),cbr(512,512),cbr(512,512))
        self.d4 = nn.Sequential(cbr(512,512),cbr(512,512),cbr(512,256))
        self.d3 = nn.Sequential(cbr(256,256),cbr(256,256),cbr(256,128))
        self.d2 = nn.Sequential(cbr(128,128),cbr(128,64))
        self.d1 = nn.Sequential(cbr(64,64), nn.Conv2d(64,oc,1))

    def forward(self, x):
        x,i1=self.pool(self.e1(x)); x,i2=self.pool(self.e2(x))
        x,i3=self.pool(self.e3(x)); x,i4=self.pool(self.e4(x))
        x,i5=self.pool(self.e5(x))
        x=self.d5(self.unpool(x,i5)); x=self.d4(self.unpool(x,i4))
        x=self.d3(self.unpool(x,i3)); x=self.d2(self.unpool(x,i2))
        return self.d1(self.unpool(x,i1))


# ══════════════════════════════════════════════════════════════════
# TRENING DL MODELA (zajednička funkcija za U-Net i SegNet)
# ══════════════════════════════════════════════════════════════════

def train_dl_model(model, model_name):
    print(f"\n{'═'*55}")
    print(f"TRENIRANJE — {model_name}")
    print(f"{'═'*55}")

    train_dl, val_dl, test_dl, test_idx, img_paths, mask_paths = \
        build_dl_loaders()

    save_path = os.path.join(CFG["results_dir"], f"{model_name.lower()}_best.pth")
    criterion = FocalDiceLoss(pos_weight=4.0)
    optimizer = optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["num_epochs"])

    best_dice = 0.0
    history   = []

    print(f"\n{'Epoch':>5} | {'Train L':>7} | {'Val Acc':>7} | {'Val Dice':>8} | {'Val IoU':>7}")
    print("─" * 48)

    for epoch in range(1, CFG["num_epochs"]+1):
        model.train()
        tl = 0
        for imgs, masks in tqdm(train_dl, desc=f"Ep {epoch:3d}",
                                 leave=False, ncols=65):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item()
        tl /= len(train_dl)

        val_acc, val_dice, val_iou = evaluate_dl(model, val_dl)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": round(tl,4),
                        "val_acc": round(val_acc,4),
                        "val_dice": round(val_dice,4)})

        print(f"{epoch:>5d} | {tl:>7.4f} | {val_acc:>7.4f} | "
              f"{val_dice:>8.4f} | {val_iou:>7.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), save_path)
            print(f"        ✓ best model (Dice={best_dice:.4f})")

    # Testiranje
    print(f"\n{'─'*48}")
    print(f"TESTNI SKUP — {model_name}")
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    test_acc, test_dice, test_iou = evaluate_dl(model, test_dl)
    print(f"  Pixel Accuracy  : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"  Dice koeficijent: {test_dice:.4f}")
    print(f"  IoU             : {test_iou:.4f}")

    # Spremi history
    hist_path = os.path.join(CFG["results_dir"], f"{model_name.lower()}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    # Vizualizacija predikcija
    _save_dl_predictions(model, model_name, test_idx, img_paths, mask_paths)

    return {"pixel_accuracy": round(test_acc,4),
            "dice_coefficient": round(test_dice,4),
            "iou_score": round(test_iou,4)}


@torch.no_grad()
def _save_dl_predictions(model, name, test_idx, img_paths, mask_paths, n=4):
    model.eval()
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    sz   = CFG["img_size"]
    rows = []

    for idx in random.sample(test_idx, min(n, len(test_idx))):
        img_pil  = Image.open(img_paths[idx]).convert("RGB")
        mask_pil = Image.open(mask_paths[idx]).convert("L")
        img_r  = img_pil.resize((sz,sz), Image.LANCZOS)
        mask_r = mask_pil.resize((sz,sz), Image.NEAREST)

        img_t  = TF.normalize(TF.to_tensor(img_r),
                              [0.485,0.456,0.406],[0.229,0.224,0.225])
        mask_t = (TF.to_tensor(mask_r) > 0.5).float().squeeze()
        pred   = (torch.sigmoid(model(img_t.unsqueeze(0).to(DEVICE)))
                  ).squeeze().cpu()
        pred_b = (pred > 0.5).float()

        img_np   = (TF.to_tensor(img_r).permute(1,2,0).numpy()*255).astype(np.uint8)
        mask_np  = (mask_t.numpy()*255).astype(np.uint8)
        pred_np  = (pred_b.numpy()*255).astype(np.uint8)
        overlay  = img_np.copy()
        overlay[pred_np>127] = (overlay[pred_np>127]*0.45 +
                                 np.array([220,50,50])*0.55).clip(0,255).astype(np.uint8)

        acc  = pixel_accuracy(pred_b, mask_t)
        dice = dice_coefficient(pred_b, mask_t)
        row  = np.concatenate(
            [img_np, np.stack([mask_np]*3,-1),
             np.stack([pred_np]*3,-1), overlay], axis=1)
        row_pil = Image.fromarray(row)
        ImageDraw.Draw(row_pil).text(
            (4,4), f"Acc={acc:.3f}  Dice={dice:.3f}", fill=(255,255,0))
        rows.append(np.array(row_pil))

    hdr    = _make_header(sz*4, ["Original","GT Maska","Predikcija","Overlay"])
    canvas = np.concatenate([hdr]+rows, axis=0)
    path   = os.path.join(CFG["results_dir"], f"{name.lower()}_predictions.png")
    Image.fromarray(canvas).save(path)
    print(f"  Vizualizacija → {path}")


# ══════════════════════════════════════════════════════════════════
# RANDOM FOREST
# ══════════════════════════════════════════════════════════════════

def _extract_features(img_bgr):
    """
    Ručno definirane značajke po pikselu:
      - Boja: RGB, HSV, LAB  (9 značajki)
      - Tekstura: Gaussov blur na 3 razine (3)
      - Gradijent: Sobelova magnituda (1)
      - Lokalna varijansa (1)
      - Gabor filteri: 8 orijentacija × 2 frekvencije (16)
    Ukupno: ~30 značajki po pikselu
    """
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,0]/=180.; hsv[:,:,1:]/=255.
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    lab[:,:,0]/=100.; lab[:,:,1:]=(lab[:,:,1:]+128.)/255.

    blurs = [cv2.GaussianBlur(g,(0,0),s)[:,:,None] for s in [1,3,5]]
    gx = cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3)
    gy = cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)
    mag = np.sqrt(gx**2+gy**2)[:,:,None]
    mn  = cv2.blur(g,(5,5)); sq = cv2.blur(g**2,(5,5))
    var = (sq-mn**2).clip(0)[:,:,None]
    gabors = []
    for th in np.linspace(0, np.pi, 8, endpoint=False):
        for lm in [5,10]:
            k = cv2.getGaborKernel((11,11),3.,th,lm,.5,0,cv2.CV_32F)
            gabors.append(cv2.filter2D(g,cv2.CV_32F,k)[:,:,None])
    H,W = g.shape
    return np.concatenate([rgb,hsv,lab]+blurs+[mag,var]+gabors,
                          axis=-1).reshape(H*W,-1).astype(np.float32)


def train_rf_model():
    print(f"\n{'═'*55}")
    print("TRENIRANJE — Random Forest")
    print(f"{'═'*55}")

    img_paths, mask_paths = _get_paths()
    n = len(img_paths)
    train_idx, val_idx, test_idx = _split_indices(n)
    sz = CFG["rf_img_size"]

    def load_data(indices, max_imgs, ratio, desc):
        X, y = [], []
        for i in tqdm(indices[:max_imgs], desc=desc):
            img  = np.array(Image.open(img_paths[i]).convert("RGB")
                            .resize((sz,sz), Image.LANCZOS))
            mask = np.array(Image.open(mask_paths[i]).convert("L")
                            .resize((sz,sz), Image.NEAREST))
            bgr  = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            mb   = (mask>127).astype(np.uint8).ravel()
            feat = _extract_features(bgr)
            n_px = feat.shape[0]
            n_sel= max(1, int(n_px*ratio))
            fg   = np.where(mb==1)[0]; bg = np.where(mb==0)[0]
            nfg  = min(n_sel//2, len(fg)); nbg = min(n_sel-nfg, len(bg))
            ch   = np.concatenate([np.random.choice(fg,nfg,replace=False),
                                   np.random.choice(bg,nbg,replace=False)])
            X.append(feat[ch]); y.append(mb[ch])
        return np.vstack(X), np.concatenate(y)

    X_tr, y_tr = load_data(train_idx, CFG["rf_max_train"],
                            CFG["rf_pixel_ratio"], "RF train")

    print(f"\nTreniram RF (n={CFG['rf_n_estimators']}, "
          f"depth={CFG['rf_max_depth']})...")
    rf = RandomForestClassifier(
        n_estimators=CFG["rf_n_estimators"],
        max_depth=CFG["rf_max_depth"],
        n_jobs=-1, random_state=CFG["random_seed"],
        class_weight="balanced", verbose=0)
    rf.fit(X_tr, y_tr)

    save_path = os.path.join(CFG["results_dir"], "rf_model.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(rf, f)
    print(f"✓ RF model → {save_path}")

    # Testiranje
    print("\nEvaluacija RF na testnom skupu...")
    acc_l, dice_l, iou_l = [], [], []
    for i in tqdm(test_idx[:200], desc="RF test"):
        img  = np.array(Image.open(img_paths[i]).convert("RGB")
                        .resize((sz,sz), Image.LANCZOS))
        mask = np.array(Image.open(mask_paths[i]).convert("L")
                        .resize((sz,sz), Image.NEAREST))
        bgr  = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        mb   = (mask>127).astype(np.uint8).ravel()
        pred = rf.predict(_extract_features(bgr))
        acc_l.append((pred==mb).mean())
        dice_l.append(f1_score(mb, pred, zero_division=0))
        iou_l.append(jaccard_score(mb, pred, zero_division=0))

    ta, td, ti = np.mean(acc_l), np.mean(dice_l), np.mean(iou_l)
    print(f"\n  Pixel Accuracy  : {ta:.4f}  ({ta*100:.2f}%)")
    print(f"  Dice koeficijent: {td:.4f}")
    print(f"  IoU             : {ti:.4f}")

    _save_rf_predictions(rf, test_idx, img_paths, mask_paths)
    return {"pixel_accuracy": round(ta,4),
            "dice_coefficient": round(td,4),
            "iou_score": round(ti,4)}


def _save_rf_predictions(rf, test_idx, img_paths, mask_paths, n=4):
    sz  = CFG["rf_img_size"]
    rows = []
    for idx in random.sample(test_idx, min(n, len(test_idx))):
        img  = np.array(Image.open(img_paths[idx]).convert("RGB")
                        .resize((sz,sz), Image.LANCZOS))
        mask = np.array(Image.open(mask_paths[idx]).convert("L")
                        .resize((sz,sz), Image.NEAREST))
        bgr  = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        mb   = (mask>127).astype(np.uint8)
        pred = rf.predict(_extract_features(bgr)).reshape(sz,sz).astype(np.uint8)

        mask_np  = (mb*255).astype(np.uint8)
        pred_np  = (pred*255).astype(np.uint8)
        overlay  = img.copy()
        overlay[pred>0] = (overlay[pred>0]*0.45 +
                           np.array([220,50,50])*0.55).clip(0,255).astype(np.uint8)

        acc  = (pred.ravel()==mb.ravel()).mean()
        dice = f1_score(mb.ravel(), pred.ravel(), zero_division=0)
        row  = np.concatenate(
            [img, np.stack([mask_np]*3,-1),
             np.stack([pred_np]*3,-1), overlay], axis=1)
        row_pil = Image.fromarray(row)
        ImageDraw.Draw(row_pil).text(
            (4,4), f"Acc={acc:.3f}  Dice={dice:.3f}", fill=(255,255,0))
        rows.append(np.array(row_pil))

    hdr    = _make_header(sz*4, ["Original","GT Maska","Predikcija","Overlay"])
    canvas = np.concatenate([hdr]+rows, axis=0)
    path   = os.path.join(CFG["results_dir"], "rf_predictions.png")
    Image.fromarray(canvas).save(path)
    print(f"  Vizualizacija → {path}")


# ══════════════════════════════════════════════════════════════════
# USPOREDBA — tablica i grid vizualizacija
# ══════════════════════════════════════════════════════════════════

def _make_header(width, labels, height=26):
    hdr = np.zeros((height, width, 3), dtype=np.uint8)
    pil = Image.fromarray(hdr)
    draw = ImageDraw.Draw(pil)
    col_w = width // len(labels)
    for i, lbl in enumerate(labels):
        draw.text((i*col_w+5, 6), lbl, fill=(255,255,255))
    return np.array(pil)


def save_comparison_table(results: dict):
    """Spremi usporednu tablicu metrika."""
    line = "─" * 62
    header = (f"\n{'═'*62}\n"
              f"  USPOREDBA MODELA SEGMENTACIJE — HAM10000\n"
              f"{'═'*62}\n"
              f"  {'Model':<20} {'Pixel Accuracy':>15} {'Dice koef.':>12} {'IoU':>8}\n"
              f"{line}")
    rows = ""
    for name, r in results.items():
        rows += (f"\n  {name:<20} {r['pixel_accuracy']:>15.4f} "
                 f"{r['dice_coefficient']:>12.4f} {r['iou_score']:>8.4f}")
    best_dice  = max(results, key=lambda k: results[k]['dice_coefficient'])
    best_acc   = max(results, key=lambda k: results[k]['pixel_accuracy'])
    footer = (f"\n{line}\n"
              f"  Najbolji Dice:           {best_dice}\n"
              f"  Najbolji Pixel Accuracy: {best_acc}\n"
              f"{'═'*62}\n\n"
              f"Metrike:\n"
              f"  Pixel Accuracy   = (TP+TN) / ukupno piksela\n"
              f"                     Udio ispravno klasificiranih piksela\n"
              f"  Dice koeficijent = 2·|X∩Y| / (|X|+|Y|)\n"
              f"                     Mjera preklapanja; otporan na neuravnoteženost klasa\n"
              f"  IoU              = |X∩Y| / |X∪Y|  (Jaccard indeks)\n")

    table = header + rows + footer
    print(table)
    path = os.path.join(CFG["results_dir"], "comparison_table.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(table)
    print(f"✓ Tablica → {path}")


def run_comparison():
    """Učitaj sve rezultate i ispiši usporedbu."""
    metrics_path = os.path.join(CFG["results_dir"], "metrics.json")
    if not os.path.exists(metrics_path):
        print("⚠ metrics.json ne postoji. Pokrenite sve faze treniranja.")
        return
    with open(metrics_path) as f:
        results = json.load(f)
    save_comparison_table(results)


# ══════════════════════════════════════════════════════════════════
# GLAVNI PIPELINE
# ══════════════════════════════════════════════════════════════════

def run_all():
    all_metrics = {}

    print("\n" + "▶" * 3 + " FAZA 1: PREDOBRADA")
    run_preprocessing()

    print("\n" + "▶" * 3 + " FAZA 2: U-NET")
    unet = UNet().to(DEVICE)
    all_metrics["U-Net"] = train_dl_model(unet, "UNet")

    print("\n" + "▶" * 3 + " FAZA 3: SEGNET")
    segnet = SegNet().to(DEVICE)
    all_metrics["SegNet"] = train_dl_model(segnet, "SegNet")

    print("\n" + "▶" * 3 + " FAZA 4: RANDOM FOREST")
    all_metrics["Random Forest"] = train_rf_model()

    # Spremi sve metrike
    metrics_path = os.path.join(CFG["results_dir"], "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n✓ Sve metrike → {metrics_path}")

    print("\n" + "▶" * 3 + " USPOREDBA")
    save_comparison_table(all_metrics)

    print("\n" + "═"*55)
    print("PIPELINE ZAVRŠEN — svi rezultati su u mapi results/")
    print("═"*55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all",
        choices=["all","preprocess","train_unet",
                 "train_segnet","train_rf","compare"])
    args = parser.parse_args()

    if args.phase == "all":
        run_all()
    elif args.phase == "preprocess":
        run_preprocessing()
    elif args.phase == "train_unet":
        unet = UNet().to(DEVICE)
        m = train_dl_model(unet, "UNet")
        _update_metrics("U-Net", m)
    elif args.phase == "train_segnet":
        segnet = SegNet().to(DEVICE)
        m = train_dl_model(segnet, "SegNet")
        _update_metrics("SegNet", m)
    elif args.phase == "train_rf":
        m = train_rf_model()
        _update_metrics("Random Forest", m)
    elif args.phase == "compare":
        run_comparison()


def _update_metrics(name, metrics):
    path = os.path.join(CFG["results_dir"], "metrics.json")
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    data[name] = metrics
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✓ Metrike ažurirane → {path}")
