"""Paper §3.1.2 / §3.3.2 losses. Equations (5)–(11)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

TAU = 0.0  # §3.3, margin τ


def dice_loss(prob: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    gt = gt.float()
    inter = (prob * gt).sum()
    return 1.0 - 2.0 * inter / (prob.sum() + gt.sum() + eps)


def ce_loss(prob: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy(prob.clamp(1e-6, 1.0 - 1e-6), gt.float())


def contrast_loss(central_tokens: torch.Tensor, tau: float = TAU) -> torch.Tensor:
    """Eq. (5). Central tokens are distinct patterns → y = −1, ℒ = max(0, cos − τ)."""
    t = central_tokens.reshape(-1, central_tokens.shape[-1])
    k = t.shape[0]
    loss = t.new_zeros(())
    n_pairs = 0
    for i in range(k):
        for j in range(i + 1, k):
            cos = F.cosine_similarity(t[i : i + 1], t[j : j + 1]).squeeze()
            loss = loss + torch.relu(cos - tau)
            n_pairs += 1
    return loss / max(n_pairs, 1)


def elbo_loss(mu: torch.Tensor, sigma: torch.Tensor, recon_nll: torch.Tensor) -> torch.Tensor:
    """Eq. (6)–(7): ℒ_ELBO = −(𝔼 log p(x|z) − D_KL), prior 𝒩(0, 1)."""
    sigma = sigma.clamp_min(1e-8)
    kl = 0.5 * (torch.log(1.0 / sigma.pow(2)) + (sigma.pow(2) + mu.pow(2)) / 1.0 - 1.0)
    return recon_nll + kl.mean()


def unw_loss(u: torch.Tensor, mu_x, sigma_x, mu_c, sigma_c) -> torch.Tensor:
    """Eq. (8): ℒ_UnW = U · [log(σx²/σc²) + (σc² + (μx−μc)²)/σx² − 1]."""
    sigma_x = sigma_x.clamp_min(1e-8)
    sigma_c = sigma_c.clamp_min(1e-8)
    gap = torch.log(sigma_x.pow(2) / sigma_c.pow(2)) + (sigma_c.pow(2) + (mu_x - mu_c).pow(2)) / sigma_x.pow(2) - 1.0
    return u * gap.mean()


def corr_loss(d_kl: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Eq. (9): (1 − Pearson(D_KL, U)) + (1 − cos(D_KL, U))."""
    d = d_kl.reshape(-1).float()
    v = u.reshape(-1).float()
    if d.numel() < 2:
        return d.new_zeros(())
    d0, v0 = d - d.mean(), v - v.mean()
    pearson = (d0 * v0).sum() / (d0.norm() * v0.norm() + 1e-8)
    cos = F.cosine_similarity(d[None], v[None])
    return (1.0 - pearson) + (1.0 - cos)


def uncertainty_map(samples: torch.Tensor):
    """Eq. (3)–(4). samples: (B, N, H, W) in [0, 1], N = 20."""
    mean = samples.mean(dim=1, keepdim=True)
    u_map = torch.sqrt(((samples - mean) ** 2).mean(dim=1, keepdim=True) + 1e-12)
    u = u_map.sum() / (mean.sum() + 1e-8)
    return mean, u_map, u


def distributional_gap(mu_x, sigma_x, mu_c, sigma_c) -> torch.Tensor:
    """D_KL used as the gap scalar (mean over token dims)."""
    sigma_x = sigma_x.clamp_min(1e-8)
    sigma_c = sigma_c.clamp_min(1e-8)
    kl = torch.log(sigma_c / sigma_x) + (sigma_x.pow(2) + (mu_x - mu_c).pow(2)) / (2.0 * sigma_c.pow(2)) - 0.5
    return kl.mean()


def total_loss(dice, ce, contrast, elbo, unw, corr) -> torch.Tensor:
    """Eq. (11), all weights = 1."""
    return dice + ce + contrast + elbo + unw + corr
