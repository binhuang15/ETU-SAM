#!/usr/bin/env python3
"""Interactive / CLI inference: segmentation + uncertainty map."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from etu_sam.model import ETUSAM, forward_etu, load_checkpoint
from etu_sam.preprocess import DataPreprocessor
from etu_sam.seed import set_seed

IMAGE_TYPES = [("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")]
MASK_TYPES = [("Masks", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")]


def pick_file(title: str, filetypes) -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.update()
        path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        root.destroy()
        return str(path or "")
    except Exception:
        return input(f"{title}: ").strip()


def read_image(path: str) -> np.ndarray:
    img = np.asarray(Image.open(path).convert("L"))
    if img.ndim < 3:
        img = np.repeat(img[..., None], 3, 2)
    return img


def read_mask(path: str) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L"))
    if mask.ndim > 2:
        mask = mask[..., 0]
    return mask


def overlay_mask(image_rgb: np.ndarray, binary: np.ndarray, color=(0, 180, 80), alpha=0.45) -> np.ndarray:
    out = image_rgb.astype(np.float32)
    tint = np.zeros_like(out)
    tint[..., 0], tint[..., 1], tint[..., 2] = color
    m = binary.astype(bool)
    out[m] = (1.0 - alpha) * out[m] + alpha * tint[m]
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="ETU-SAM inference: choose an image, save mask + uncertainty map.")
    ap.add_argument("--image", default="", help="Input image. If omitted, a file dialog opens.")
    ap.add_argument("--mask", default="", help="Optional GT mask used only to build the box prompt (+15 px).")
    ap.add_argument("--box", nargs=4, type=float, default=None, metavar=("X1", "Y1", "X2", "Y2"),
                    help="Box in original-image pixels. Used if --mask is not given.")
    ap.add_argument("--ckpt", default="checkpoints/ETU-SAM.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-show", action="store_true", help="Save files only, do not open a window.")
    args = ap.parse_args()

    set_seed(args.seed)
    image_path = args.image or pick_file("Select an ultrasound image", IMAGE_TYPES)
    if not image_path:
        raise SystemExit("No image selected.")
    image_path = str(Path(image_path).expanduser().resolve())
    img = read_image(image_path)
    orig_hw = img.shape[:2]

    mask_path = args.mask
    if not mask_path and args.box is None and not args.image:
        mask_path = pick_file("Select a GT mask for the box prompt (Cancel = full-image box)", MASK_TYPES)

    pre = DataPreprocessor(image_size=256, bbox_shift=15)
    if mask_path:
        mask = read_mask(mask_path)
        batch = pre.preprocess(img, mask, box_jitter=False, box_expand=15, sample_instance=False)
        x = batch["image"].unsqueeze(0)
        box = batch["bboxes"].unsqueeze(0)
        prompt_note = f"GT box +15 px from {Path(mask_path).name}"
    elif args.box is not None:
        tensor, _ = pre.encode_image(img)
        x = tensor.unsqueeze(0)
        box = pre.transform_box(args.box, orig_hw, expand=15)
        prompt_note = f"box {args.box} (+15 px)"
    else:
        tensor, _ = pre.encode_image(img)
        x = tensor.unsqueeze(0)
        h, w = orig_hw
        box = pre.transform_box((0, 0, w - 1, h - 1), orig_hw, expand=0)
        prompt_note = "full-image box (paper protocol uses a GT box +15 px)"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = ETUSAM()
    missing, unexpected = load_checkpoint(model, args.ckpt, strict=True)
    print(f"loaded {args.ckpt} missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device).eval()
    x = x.to(device)
    box = box.to(device)

    _, packed = forward_etu(model, x, box)
    prob = packed["prob"][0, 0].cpu().numpy()
    u_map = packed["u_map"][0, 0].cpu().numpy()
    pred = (prob > 0.5).astype(np.uint8)
    u_scalar = float(packed["U"].cpu())

    pred_orig = pre.restore(pred.astype(np.float32), orig_hw, nearest=True) > 0.5
    u_orig = pre.restore(u_map.astype(np.float32), orig_hw, nearest=False)
    gray = img[..., 0]
    vis = np.stack([gray, gray, gray], axis=-1)
    overlay = overlay_mask(vis, pred_orig)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    mask_png = out_dir / f"{stem}_mask.png"
    u_png = out_dir / f"{stem}_uncertainty.png"
    panel_png = out_dir / f"{stem}_panel.png"
    Image.fromarray((pred_orig.astype(np.uint8) * 255)).save(mask_png)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True)
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Input")
    axes[1].imshow(overlay)
    axes[1].set_title("Segmentation")
    im = axes[2].imshow(u_orig, cmap="inferno")
    axes[2].set_title(rf"$U_{{\mathrm{{map}}}}$  ($U$={u_scalar:.4f})")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    fig.suptitle(f"{Path(image_path).name}  |  {prompt_note}", fontsize=10)
    fig.savefig(panel_png, dpi=150)

    u_norm = u_orig / (u_orig.max() + 1e-8)
    cmap = plt.get_cmap("inferno")
    u_rgb = (cmap(u_norm)[..., :3] * 255).astype(np.uint8)
    Image.fromarray(u_rgb).save(u_png)

    print(f"image:        {image_path}")
    print(f"prompt:       {prompt_note}")
    print(f"U (Eq. 4):    {u_scalar:.6f}")
    print(f"mask:         {mask_png}")
    print(f"uncertainty:  {u_png}")
    print(f"panel:        {panel_png}")

    if not args.no_show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)


if __name__ == "__main__":
    main()
