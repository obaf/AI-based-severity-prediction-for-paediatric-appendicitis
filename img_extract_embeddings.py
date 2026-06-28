"""
Extract per-image embeddings from the Regensburg pediatric appendicitis US images
using a modern FROZEN vision foundation model.

Primary encoder: BiomedCLIP (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
  -- a biomedical vision-language foundation model (domain-appropriate, modern).

Output: image_embeddings_<tag>.npz with arrays:
    emb     (N_images, D)  float32
    subject (N_images,)    int    -- US_Number parsed from filename
    fname   (N_images,)    str
"""
import warnings; warnings.filterwarnings("ignore")
import os, re, sys, glob, zipfile
import numpy as np
import torch
from PIL import Image

TAG = sys.argv[1] if len(sys.argv) > 1 else "biomedclip"
ZIP = "US_Pictures.zip"
IMG_DIR = "US_Pictures"

# ---- 1. Unzip if needed ----------------------------------------------------
if not os.path.isdir(IMG_DIR) or len(glob.glob(os.path.join(IMG_DIR, "**", "*.bmp"), recursive=True)) == 0:
    print("Unzipping", ZIP, "...")
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(".")
    print("Unzipped.")

bmps = glob.glob(os.path.join(IMG_DIR, "**", "*.bmp"), recursive=True)
print(f"Found {len(bmps)} .bmp images")

# Parse subject number: filename starts with "<subject>.<view> ..."
def subject_of(path):
    base = os.path.basename(path)
    m = re.match(r"^(\d+)\.", base)
    return int(m.group(1)) if m else None

pairs = [(p, subject_of(p)) for p in bmps]
pairs = [(p, s) for p, s in pairs if s is not None]
print(f"Parsed subject id for {len(pairs)} images; "
      f"{len(set(s for _, s in pairs))} unique subjects")

# ---- 2. Load frozen encoder ------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(os.cpu_count())
print("Device:", device, "| threads:", torch.get_num_threads())

if TAG == "biomedclip":
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    model = model.to(device).eval()
    def embed(imgs):  # imgs: list of PIL
        x = torch.stack([preprocess(im) for im in imgs]).to(device)
        with torch.no_grad():
            f = model.encode_image(x)
        return f.cpu().numpy()
elif TAG == "dinov2":
    import timm
    model = timm.create_model("vit_base_patch14_dinov2.lvd142m",
                              pretrained=True, num_classes=0, img_size=224).to(device).eval()
    cfg = timm.data.resolve_data_config({"input_size": (3, 224, 224)}, model=model)
    transform = timm.data.create_transform(**cfg)
    print("DINOv2 transform input_size:", cfg.get("input_size"))
    def embed(imgs):
        x = torch.stack([transform(im) for im in imgs]).to(device)
        with torch.no_grad():
            f = model(x)
        return f.cpu().numpy()
else:
    raise ValueError(TAG)

# ---- 3. Extract in batches -------------------------------------------------
embs, subs, fnames = [], [], []
BATCH = 32
buf_imgs, buf_meta = [], []

def flush():
    if not buf_imgs:
        return
    e = embed(buf_imgs)
    embs.append(e)
    for (p, s) in buf_meta:
        subs.append(s); fnames.append(os.path.basename(p))
    buf_imgs.clear(); buf_meta.clear()

for i, (p, s) in enumerate(pairs):
    try:
        im = Image.open(p).convert("RGB")
    except Exception as ex:
        print("skip", p, ex); continue
    buf_imgs.append(im); buf_meta.append((p, s))
    if len(buf_imgs) >= BATCH:
        flush()
        if (i + 1) % 320 == 0:
            print(f"  {i+1}/{len(pairs)} images embedded")
flush()

emb = np.concatenate(embs, axis=0).astype(np.float32)
subject = np.array(subs, dtype=int)
fname = np.array(fnames, dtype=object)
print("Embeddings:", emb.shape)
np.savez(f"image_embeddings_{TAG}.npz", emb=emb, subject=subject, fname=fname)
print(f"Saved image_embeddings_{TAG}.npz")
