"""Mask decoder with the D2U estimator (central tokens Tc + DTS)."""
from __future__ import annotations

import copy
from typing import List, Tuple, Type

import torch
from torch import nn
from torch.nn import functional as F

from .common import LayerNorm2d


class D2UMaskDecoder(nn.Module):
    """SAM mask decoder plus D2U: central tokens, UW-BML distributions, and DTS (Eq. 10)."""

    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        dts_samples: int = 20,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.dts_samples = max(1, int(dts_samples))
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        hyper_out = 2 * transformer_dim // 8
        self.output_hypernetworks_mlps = nn.ModuleList(
            [MLP(transformer_dim, transformer_dim, hyper_out, 3) for _ in range(self.num_mask_tokens)]
        )
        self.iou_prediction_head = MLP(transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth)
        # Paper Tc; checkpoint key remains mid_tokens.
        self.mid_tokens = nn.Embedding(self.num_mask_tokens, 256)
        self.mu_c: List = [None] * self.num_mask_tokens
        self.sigma_c: List = [None] * self.num_mask_tokens
        self.transformer2 = copy.deepcopy(transformer)
        self.output_hypernetworks_mlps2 = copy.deepcopy(self.output_hypernetworks_mlps)

    @property
    def central_tokens(self) -> nn.Embedding:
        return self.mid_tokens

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        distributional_gap_feat: torch.Tensor,
        multimask_output: bool,
        uncertainty_output: bool,
    ):
        masks, iou_pred, mu_x, sigma_x, cos_tokens = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            distributional_gap_feat=distributional_gap_feat,
        )
        samples = torch.sigmoid(masks)
        prob = samples.mean(dim=1, keepdim=True)
        iou = iou_pred.mean(dim=1, keepdim=True)
        return prob, iou, samples, mu_x, sigma_x, cos_tokens

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        distributional_gap_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        src = image_embeddings + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        decoder = self.transformer2 if distributional_gap_feat is not None else self.transformer
        hs, src = decoder(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]
        if mask_tokens_out.shape[0] != 1:
            raise RuntimeError("ETU-SAM uses batch size 1 (paper §3.3).")
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)

        t_c = self.central_tokens.weight.reshape(-1, self.num_mask_tokens, 256)
        cos_tokens = F.cosine_similarity(t_c, mask_tokens_out, dim=-1)
        max_tokens = int(torch.argmax(cos_tokens).item())
        hyper_mlps = (
            self.output_hypernetworks_mlps2
            if distributional_gap_feat is not None
            else self.output_hypernetworks_mlps
        )

        mu_sigma_x = hyper_mlps[max_tokens](mask_tokens_out[:, max_tokens, :])
        half = mu_sigma_x.shape[1] // 2
        mu_x = mu_sigma_x[:, :half]
        sigma_x = torch.exp(mu_sigma_x[:, half:])

        mu_sigma_c = hyper_mlps[max_tokens](t_c[:, max_tokens, :])
        self.mu_c[max_tokens] = mu_sigma_c[0, :half]
        self.sigma_c[max_tokens] = torch.exp(mu_sigma_c[0, half:])

        kernels = torch.stack([self.dts(mu_x, sigma_x) for _ in range(self.dts_samples)], dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (kernels @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        masks = F.interpolate(masks, (256, 256), mode="bilinear", align_corners=False)
        iou_pred = self.iou_prediction_head(iou_token_out)
        return masks, iou_pred, mu_x, sigma_x, cos_tokens

    @staticmethod
    def dts(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """DTS, Eq. (10): Tx = mu + sigma * tanh(eps), eps ~ N(0, I)."""
        eps = torch.tanh(torch.randn_like(sigma))
        return mu + eps * sigma


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, sigmoid_output: bool = False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x
