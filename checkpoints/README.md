# Checkpoints

Place the official ETU-SAM weights here:

```
checkpoints/ETU-SAM.pt
```

The file is ~367 MB; Git LFS is recommended (`git lfs track "checkpoints/*.pt"`).

Download SAM-Med2D ViT-B (`sam-med2d_b.pth`) separately if you train from scratch; pass it to `train.py --pretrained`.
