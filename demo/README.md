Public ultrasound demos used with `infer.py`. Masks are only for the gold-standard box prompt (+15 px).

| File | Dataset | Anatomy |
|---|---|---|
| `UDIAT_000129.png` | UDIAT (Yap et al.) | Breast |
| `HC18_362.png` | HC18 (van den Heuvel et al.) | Fetal head |
| `CCA_slice99.png` | CCA (Momot) | Carotid |

```bash
python infer.py --image demo/UDIAT_000129.png --mask demo/UDIAT_000129_mask.png
```
