#!/usr/bin/env python3
"""Train ETU-SAM as specified in paper §3.3."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from etu_sam.augment import paper_transforms
from etu_sam.losses import ce_loss, contrast_loss, corr_loss, dice_loss, elbo_loss, total_loss, unw_loss
from etu_sam.model import (
    ETUSAM,
    attach_central_tokens,
    build_uncertainty_prompt,
    freeze_image_encoder,
    load_ftsam_init,
    pack_d2u,
)
from etu_sam.preprocess import DataPreprocessor
from etu_sam.seed import set_seed

EVAL_INTERVAL = 500  # §3.3: training loss evaluated every 500 iterations


def load_gray_rgb_mask(image_path: str, mask_path: str):
    img = np.asarray(Image.open(image_path).convert("L"))
    if img.ndim < 3:
        img = np.repeat(img[..., None], 3, 2)
    mask = np.asarray(Image.open(mask_path).convert("L"))
    if mask.ndim > 2:
        mask = mask[..., 0]
    return img, mask


def paper_losses(packed, gt, d_kl_hist, u_hist):
    recon = ce_loss(packed["samples"], gt.expand_as(packed["samples"]))
    l_dice = dice_loss(packed["prob"], gt)
    l_ce = ce_loss(packed["prob"], gt)
    l_contrast = contrast_loss(packed["central_tokens"])
    l_elbo = elbo_loss(packed["mu"], packed["sigma"], recon)
    l_unw = unw_loss(packed["U"], packed["mu"], packed["sigma"], packed["mu_c"], packed["sigma_c"])
    # Keep the current sample in the graph so Eq. (9) has a gradient (batch size is 1).
    d_now = packed["d_kl"].reshape(())
    u_now = packed["U"].reshape(())
    l_corr = packed["prob"].new_zeros(())
    if d_kl_hist:
        l_corr = corr_loss(torch.stack(d_kl_hist + [d_now]), torch.stack(u_hist + [u_now]))
    d_kl_hist.append(d_now.detach())
    u_hist.append(u_now.detach())
    return total_loss(l_dice, l_ce, l_contrast, l_elbo, l_unw, l_corr)


def run_pass(model, x, box, gt, d_kl_hist, u_hist, mask_prompt=None, u_map=None, distributional_gap_feat=None):
    packed = attach_central_tokens(
        model,
        pack_d2u(
            model(
                x,
                boxes=box,
                mask_prompt=mask_prompt,
                u_map=u_map,
                distributional_gap_feat=distributional_gap_feat,
                uncertainty_output=True,
            )
        ),
    )
    gap_feat, u_map_out = build_uncertainty_prompt(model, packed)
    loss = paper_losses(packed, gt, d_kl_hist, u_hist)
    return packed, loss, gap_feat, u_map_out


def main():
    ap = argparse.ArgumentParser(description="Train ETU-SAM (paper §3.3).")
    ap.add_argument("--data-root", required=True, help="SAM-Med2D root containing SAMed2D_v1.json")
    ap.add_argument("--pretrained", default="", help="sam-med2d_b.pth for encoder/decoder init")
    ap.add_argument("--save-dir", default="checkpoints/run")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--max-iters",
        type=int,
        default=None,
        help="Optional stop. The paper does not specify a total iteration count.",
    )
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    json_path = Path(args.data_root) / "SAMed2D_v1.json"
    with open(json_path, encoding="utf-8") as f:
        index = json.load(f)
    image_keys = list(index.keys())
    random.shuffle(image_keys)

    model = ETUSAM()
    if args.pretrained:
        load_ftsam_init(model, args.pretrained)
    freeze_image_encoder(model)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=5e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.9, patience=10, min_lr=1e-6)

    pre = DataPreprocessor(image_size=256, bbox_shift=10)
    spatial, intensity = paper_transforms(patch_size=(256, 256))

    d_kl_hist, u_hist = [], []
    running = 0.0
    n_eval = 0
    it = 0
    best = 1e9
    model.train()
    print(
        f"[ETU-SAM] eval every {EVAL_INTERVAL} iters; "
        f"max_iters={'none (paper unspecified)' if args.max_iters is None else args.max_iters}",
        flush=True,
    )
    try:
        while True:
            random.shuffle(image_keys)
            for image_index in image_keys:
                img_path = os.path.join(args.data_root, image_index)
                masks = index[image_index]
                if not masks:
                    continue
                mask_path = os.path.join(args.data_root, random.choice(masks))
                if not os.path.exists(img_path) or not os.path.exists(mask_path):
                    continue
                img, mask = load_gray_rgb_mask(img_path, mask_path)
                image = np.expand_dims(np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1)), 0)
                seg = np.expand_dims(np.expand_dims((mask > 0).astype(np.float32), 0), 0)
                data = spatial(**dict(data=image, seg=seg))
                data = intensity(**data)
                im = np.clip(np.transpose(data["data"], [0, 2, 3, 1])[0], 0, 1) * 255
                im = im.astype(np.uint8)
                mk = data["seg"][0, 0]
                batch = pre.preprocess(im, mk, box_jitter=True, sample_instance=True)
                x = batch["image"].unsqueeze(0).to(device)
                box = batch["bboxes"].unsqueeze(0).to(device)
                gt = batch["gt2D"].unsqueeze(0).float().to(device)

                opt.zero_grad()
                packed, loss1, gap_feat, u_map = run_pass(model, x, box, gt, d_kl_hist, u_hist)
                _, loss2, _, _ = run_pass(
                    model,
                    x,
                    box,
                    gt,
                    d_kl_hist,
                    u_hist,
                    mask_prompt=packed["prob"].detach(),
                    u_map=u_map,
                    distributional_gap_feat=gap_feat.detach(),
                )
                (loss1 + loss2).backward()
                opt.step()

                it += 1
                running += float(loss2.detach())
                n_eval += 1
                if it % EVAL_INTERVAL == 0:
                    avg = running / max(n_eval, 1)
                    sched.step(avg)
                    print(
                        f"[ETU-SAM] iter={it} loss={avg:.4f} lr={opt.param_groups[0]['lr']:.2e}",
                        flush=True,
                    )
                    if avg < best:
                        best = avg
                        torch.save({"sam_state_dict": model.state_dict(), "iter": it}, save_dir / "ETU-SAM_best.pt")
                    running = 0.0
                    n_eval = 0
                    d_kl_hist.clear()
                    u_hist.clear()
                    torch.save({"sam_state_dict": model.state_dict(), "iter": it}, save_dir / "ETU-SAM_last.pt")
                if args.max_iters is not None and it >= args.max_iters:
                    raise StopIteration
    except (StopIteration, KeyboardInterrupt):
        pass
    torch.save({"sam_state_dict": model.state_dict(), "iter": it}, save_dir / "ETU-SAM.pt")
    print(f"[ETU-SAM] finished iter={it} → {save_dir / 'ETU-SAM.pt'}", flush=True)


if __name__ == "__main__":
    main()
