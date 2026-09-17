# ETU-SAM

**Efficient and Transparent Uncertainty Estimation for Segment Anything Model**

Official PyTorch implementation of the method in §3.

Repository: https://github.com/binhuang15/ETU-SAM

## Requirements

```bash
pip install -r requirements.txt
```

Inputs are 256×256. Official weights: [`checkpoints/ETU-SAM.pt`](checkpoints/ETU-SAM.pt) (Git LFS).

```bash
git lfs install
git clone https://github.com/binhuang15/ETU-SAM.git
```

## Inference

```bash
python infer.py
```

A file dialog asks for an image (and optionally a GT mask to build the box prompt). Writes:

- `outputs/<name>_mask.png` — final segmentation
- `outputs/<name>_uncertainty.png` — $U_{\mathrm{map}}$ (Eq. 3)
- `outputs/<name>_panel.png` — input / overlay / uncertainty

```bash
python infer.py --image demo/UDIAT_000129.png --mask demo/UDIAT_000129_mask.png
python infer.py --image demo/HC18_362.png --mask demo/HC18_362_mask.png
python infer.py --image demo/CCA_slice99.png --mask demo/CCA_slice99_mask.png
```

Public demos (UDIAT, HC18, CCA) live in `demo/`. Without a mask or box, a full-image box is used. The paper evaluation protocol is a gold-standard box expanded by 15 pixels.

## Train

Matches **§3.3**:

- freeze image encoder; train mask decoder, D2U, and Uncertainty Transformer
- random seed **42**
- batch size 1
- AdamW $5\times 10^{-5}$, $\beta=(0.9,0.999)$, $\varepsilon=10^{-8}$, weight decay $0.01$
- ReduceLROnPlateau: factor $0.9$, patience 10 evaluations, min $10^{-6}$
- **training loss evaluated every 500 iterations**
- DTS $N=20$, central tokens $K=3$, contrastive margin $\tau=0$
- augmentation: scale $0.75$–$1.25$, rotation $\pm 45^\circ$, gamma $0.5$–$1.5$, brightness $\pm 0.1$, Gaussian noise $0.01$–$0.2$, each $p=0.5$
- train box jitter up to 10 px

The paper **does not specify** a total iteration count or gradient clipping. Use `--max-iters` only if you want an engineering stop; otherwise training continues until you interrupt it, with checkpoints written every 500 iterations.

Eq. (11), all coefficients $=1$:

$$
\mathcal{L}=\mathcal{L}_{\mathrm{Dice}}+\mathcal{L}_{\mathrm{CE}}+\mathcal{L}_{\mathrm{contrast}}+\mathcal{L}_{\mathrm{ELBO}}+\mathcal{L}_{\mathrm{UnW}}+\mathcal{L}_{\mathrm{Corr}}
$$

```bash
python train.py \
  --data-root /path/to/SAMed2Dv1 \
  --pretrained /path/to/sam-med2d_b.pth \
  --save-dir checkpoints/run \
  --device cuda:0
```

`--data-root` must contain `SAMed2D_v1.json`. The 16 ultrasound test sets are held out.

## Method map (code ↔ paper)

- **D2U** (UW-BML + DTS) → `D2UMaskDecoder`
- Central tokens $T_c$, $K=3$ → `D2UMaskDecoder.central_tokens` (checkpoint key: `mid_tokens`)
- DTS $T_x=\mu+\sigma\odot\tanh(\epsilon)$, $N=20$ → `D2UMaskDecoder.dts`
- Prompt Encoder → `PromptEncoder`
- Uncertainty Transformer → `PromptEncoder.uncertainty_transformer` (Stages 2–3)
- $U_{\mathrm{map}}$ Eq. (3), $U$ Eq. (4) → `etu_sam.losses.uncertainty_map`
- $\mathcal{L}_{\mathrm{contrast}}$ Eq. (5), $\tau=0$, $y=-1$ → `contrast_loss`
- $\mathcal{L}_{\mathrm{ELBO}}$ Eq. (6)–(7), prior $\mathcal{N}(0,1)$ → `elbo_loss`
- $\mathcal{L}_{\mathrm{UnW}}$ Eq. (8) → `unw_loss`
- $\mathcal{L}_{\mathrm{Corr}}$ Eq. (9) → `corr_loss` (Pearson+cos over the current 500-iter window; batch size is 1)
- Eq. (11) → `total_loss`
- Two-pass inference → `forward_etu`

Checkpoint parameter names (`mid_tokens`, `transformer2`, `mu_var_*`, …) are left as trained so `checkpoints/ETU-SAM.pt` still loads strictly.

## License

MIT. `segment_anything/` is adapted from [SAM](https://github.com/facebookresearch/segment-anything) and [SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D) (Apache 2.0).

## Layout

```
ETU-SAM/
  infer.py
  train.py
  demo/
  etu_sam/
  segment_anything/
  checkpoints/ETU-SAM.pt
```
