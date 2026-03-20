from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from archs.aggregation_networks import (
    GlobalWeightedFuser,
)


def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """L2-normalise embeddings along the specified dimension. Assumes batched embeddings of shape (B, D)."""
    return F.normalize(embeddings, p=2, dim=1)


def gem_pool(x: torch.Tensor, p: float = 3.0, eps: float = 1e-6) -> torch.Tensor:
    """Generalised mean pooling (https://arxiv.org/abs/1711.02512)."""
    return (
        F.avg_pool2d(x.float().clamp(min=eps).pow(p), (x.size(-2), x.size(-1)))
        .pow(1.0 / p)
        .flatten(1)
    )


# ---------------------------------------------------------------------------
# Concat embedder - naive, no learnable weights
# ---------------------------------------------------------------------------


class PoolConcatEmbedder(nn.Module):
    """
    Zero-shot embedder: pool every (timestep, scale) feature and concatenate.
    Uses features returned from LDMExtractor directly.
    """

    def __init__(
        self,
        feature_dims: dict,
        save_timesteps: list[int],
        **kwargs,
    ):
        super().__init__()

        self.embedding_dim = sum(sum(dims) for dims in feature_dims.values()) * len(
            save_timesteps
        )

    def forward(self, feats: dict, output_shape=None):
        pooled_vectors = []
        for ts in sorted(feats.keys()):
            for scale_feat in feats[ts]:
                pooled_vectors.append(gem_pool(scale_feat))

        # Concatenate all pooled vectors to get the final embedding.
        pooled_embedding = torch.cat(pooled_vectors, dim=1)

        return pooled_embedding

    def forward_unpooled(self, feats: dict, output_shape=None):
        spatial = None

        pooled_embedding = self.forward(feats, output_shape)

        # No unpooled embedding for this embedder thus None.
        return spatial, pooled_embedding


# ---------------------------------------------------------------------------
# SatDiFuser based embedders - learnable fusion weights
# ---------------------------------------------------------------------------


class FuserEmbedder(nn.Module):
    """
    Trainable embedder that uses a fuser for spatial feature transformation and aggregates them.
    """

    def __init__(
        self,
        feature_dims: dict,
        save_timesteps: list[int],
        projection_dim: int = 384,
        fuser: Literal["gwf"] = "gwf",
        **kwargs,
    ):
        self.fuser = GlobalWeightedFuser(
            feature_dims=feature_dims,
            save_timesteps=save_timesteps,
            projection_dim=projection_dim,
        )
        num_scales = len(self.fuser.scales)

        self.embedding_dim = projection_dim * num_scales

    def forward(self, feats: dict, output_shape=None):
        _, pooled_embedding = self.forward_unpooled(feats, output_shape)

        return pooled_embedding

    def forward_unpooled(self, feats: dict, output_shape=None):
        fused_feats, _ = self.fuser(feats)  # [(B, projection_dim, H_i, W_i)]

        target_h, target_w = fused_feats[0].shape[-2:]
        resized = []
        for feat in fused_feats:
            if feat.shape[-2:] != (target_h, target_w):
                feat = F.interpolate(
                    feat,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
            resized.append(feat)

        # (B, projection_dim * num_scales, H, W)
        spatial = torch.cat(resized, dim=1)

        # pool spatial features -> (B, projection_dim * num_scales)
        pooled_embedding = gem_pool(spatial)

        return spatial, pooled_embedding
