"""ETU-SAM: FT-SAM (SAM-Med2D ViT-B) + D2U + Uncertainty Transformer."""
from __future__ import annotations

from functools import partial
from pathlib import Path

import torch
import torch.nn as nn

from segment_anything.modeling.image_encoder import ImageEncoderViT
from segment_anything.modeling.mask_decoder import D2UMaskDecoder
from segment_anything.modeling.prompt_encoder import PromptEncoder
from segment_anything.modeling.transformer import TwoWayTransformer

from .losses import distributional_gap, uncertainty_map

NUM_CENTRAL_TOKENS = 3
DTS_SAMPLES = 20


class ETUSAM(nn.Module):
    def __init__(self, num_tokens: int = NUM_CENTRAL_TOKENS, dts_samples: int = DTS_SAMPLES):
        super().__init__()
        self.image_encoder = ImageEncoderViT(
            depth=12,
            embed_dim=768,
            img_size=256,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11],
            window_size=14,
            out_chans=256,
        )
        self.prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(16, 16),
            input_image_size=(256, 256),
            mask_in_chans=16,
        )
        self.mask_decoder = D2UMaskDecoder(
            num_multimask_outputs=num_tokens,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
                DAG=True,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            dts_samples=dts_samples,
        )

    def forward(
        self,
        image,
        boxes=None,
        mask_prompt=None,
        u_map=None,
        distributional_gap_feat=None,
        uncertainty_output: bool = True,
    ):
        image_embedding = self.image_encoder(image)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=mask_prompt,
            distributional_gap_feat=distributional_gap_feat,
            u_map=u_map,
            image_embed=image_embedding,
        )
        return self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            distributional_gap_feat=distributional_gap_feat,
            multimask_output=False,
            uncertainty_output=uncertainty_output,
        )


def pack_d2u(out) -> dict:
    """Pack Stage-1 D2U outputs: mean mask, U_map, U, mu_x/sigma_x."""
    prob, iou, samples, mu, sigma, cos_tokens = out
    mean, u_map, u = uncertainty_map(samples)
    return {
        "prob": mean,
        "iou": iou if iou.dim() == 2 else iou.view(iou.shape[0], -1)[:, :1],
        "samples": samples,
        "u_map": u_map,
        "U": u,
        "mu": mu,
        "sigma": sigma,
        "cos_tokens": cos_tokens,
        "central_tokens": None,
    }


def attach_central_tokens(model: ETUSAM, packed: dict) -> dict:
    """Paper Tc; weight tensor is mask_decoder.mid_tokens in the checkpoint."""
    packed["central_tokens"] = model.mask_decoder.central_tokens.weight[:NUM_CENTRAL_TOKENS]
    return packed


@torch.no_grad()
def forward_etu(model: ETUSAM, x, box):
    """Stage 1 D2U, then Stages 2–3 Uncertainty Transformer refinement."""
    packed = attach_central_tokens(model, pack_d2u(model(x, boxes=box, uncertainty_output=True)))
    gap_feat, u_map = build_uncertainty_prompt(model, packed)
    packed2 = attach_central_tokens(
        model,
        pack_d2u(
            model(
                x,
                boxes=box,
                mask_prompt=packed["prob"],
                u_map=u_map,
                distributional_gap_feat=gap_feat,
                uncertainty_output=True,
            )
        ),
    )
    build_uncertainty_prompt(model, packed2)
    return packed, packed2


def build_uncertainty_prompt(model: ETUSAM, packed: dict):
    """Pass-2 features: (mu_c, sigma_c, mu_x, sigma_x, D_KL) and U_map (paper §3.2)."""
    max_tokens = int(torch.argmax(packed["cos_tokens"]).item())
    mu_c = model.mask_decoder.mu_c[max_tokens]
    sigma_c = model.mask_decoder.sigma_c[max_tokens]
    mu_x = packed["mu"]
    sigma_x = packed["sigma"]
    d_kl = distributional_gap(mu_x, sigma_x, mu_c.unsqueeze(0), sigma_c.unsqueeze(0))
    gap_feat = torch.cat(
        [
            mu_c.unsqueeze(0).detach(),
            sigma_c.unsqueeze(0).detach(),
            mu_x.detach(),
            sigma_x.detach(),
            d_kl.detach().expand_as(mu_x),
        ],
        dim=0,
    )
    packed["d_kl"] = d_kl
    packed["mu_c"] = mu_c
    packed["sigma_c"] = sigma_c
    return gap_feat, packed["u_map"].detach()


def freeze_image_encoder(model: ETUSAM) -> None:
    """§3.3.1: freeze encoder; train decoder, D2U, and Uncertainty Transformer."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.mask_decoder.parameters():
        p.requires_grad = True
    allow = (
        "transformer_detect",
        "mlp_auto_point",
        "mlp2_auto_point",
        "norm_auto_point",
        "norm2_auto_point",
        "mu_var_self",
        "mu_var_cross",
        "mlp_mu_var",
        "mlp2_mu_var",
        "norm_mu_var",
        "norm2_mu_var",
        "uncertainty_embeddings",
        "mu_transform",
    )
    for name, p in model.prompt_encoder.named_parameters():
        p.requires_grad = any(k in name for k in allow) and "norm3_mu_var" not in name


def load_checkpoint(model: ETUSAM, path: str | Path, *, strict: bool = True):
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    state = ckpt["sam_state_dict"] if isinstance(ckpt, dict) and "sam_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=strict)
    return list(missing), list(unexpected)


def load_ftsam_init(model: ETUSAM, path: str | Path) -> None:
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model_state = model.state_dict()
    filtered = {
        k: v for k, v in state.items() if k in model_state and tuple(v.shape) == tuple(model_state[k].shape)
    }
    model.load_state_dict(filtered, strict=False)
