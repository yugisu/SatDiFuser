"""
Embedding decoders for satellite image retrieval.

Each class combines one of the three fuser backbones (Global-Weighted,
Locally-Weighted, Mixture-of-Experts) with a two-layer MLP projection
head that maps the pooled, multi-scale diffusion features to a compact,
L2-normalised embedding vector suitable for nearest-neighbour retrieval.

Usage in create_decoder (utils/tasks.py):
    task: 'embedding'
    fuser: 'gw' | 'lw' | 'moe'
"""

from torch import nn
import torch.nn.functional as F

from archs.aggregation_networks import (
    GlobalWeightedFuser,
    LocalWeightedFuser,
    MoEWeightedFuser,
)


class EmbeddingProjectorMixin:
    """
    Mixin that appends a two-layer MLP projection head on top of any fuser,
    converting the spatially-pooled multi-scale features into an (optionally
    L2-normalised) embedding vector.

    MRO expectation: must appear *before* the concrete fuser in the class
    definition, e.g.::

        class GWFuserEmbedder(EmbeddingProjectorMixin, GlobalWeightedFuser)

    The mixin forwards ``projection_dim`` to the fuser so that the bottleneck
    output channels and the projection head input dimension always agree.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        normalize_embeddings: bool = True,
        projection_dim: int = 384,  # consumed here AND forwarded to the fuser
        **kwargs,
    ):
        # Pass projection_dim down so the fuser bottlenecks output projection_dim channels.
        super().__init__(projection_dim=projection_dim, **kwargs)

        self.embedding_dim = embedding_dim
        self.normalize_embeddings = normalize_embeddings

        # Two-layer MLP: global-average-pooled features → embedding space.
        self.embedding_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, C, H, W) → (B, C, 1, 1)
            nn.Flatten(),  # → (B, C)
            nn.Linear(projection_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, embedding_dim),
        )

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, feats: dict, output_shape=None):
        """
        Args:
            feats:        pyramid dict ``{timestep: [scale_tensor, ...]}``,
                          as returned by :class:`~archs.ldm_extractor.LDMExtractor`.
            output_shape: ignored – accepted for API parity with other decoders.

        Returns:
            embedding (B, embedding_dim): L2-normalised when
                ``normalize_embeddings=True``.
            misc: whatever the underlying fuser returns as its second value
                (mixing weights, load-balancing loss, …).
        """
        fused_feats, misc = super().forward(feats)

        # fused_feats: list[(B, projection_dim, H_i, W_i)] at descending scales.
        # Upsample all to the largest spatial resolution, then sum.
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

        pooled = sum(resized)  # (B, projection_dim, H, W)
        embedding = self.embedding_head(pooled)  # (B, embedding_dim)

        if self.normalize_embeddings:
            embedding = F.normalize(embedding, p=2, dim=1)

        return embedding, misc


# ---------------------------------------------------------------------------
# Concrete embedder classes
# ---------------------------------------------------------------------------


class GWFuserEmbedder(EmbeddingProjectorMixin, GlobalWeightedFuser):
    """Global-Weighted fuser with an embedding projection head."""

    def __init__(
        self,
        feature_dims,
        projection_dim: int = 384,
        embedding_dim: int = 256,
        normalize_embeddings: bool = True,
        num_norm_groups: int = 32,
        num_res_blocks: int = 1,
        save_timesteps=None,
        num_classes=None,  # unused; accepted for create_decoder API compatibility
    ):
        super().__init__(
            embedding_dim=embedding_dim,
            normalize_embeddings=normalize_embeddings,
            projection_dim=projection_dim,
            feature_dims=feature_dims,
            num_norm_groups=num_norm_groups,
            num_res_blocks=num_res_blocks,
            save_timesteps=save_timesteps or [],
        )


class LWFuserEmbedder(EmbeddingProjectorMixin, LocalWeightedFuser):
    """Locally-Weighted fuser with an embedding projection head."""

    def __init__(
        self,
        feature_dims,
        projection_dim: int = 384,
        embedding_dim: int = 256,
        normalize_embeddings: bool = True,
        num_norm_groups: int = 32,
        num_res_blocks: int = 1,
        save_timesteps=None,
        num_classes=None,
        gating_tempature: float = 1.0,
    ):
        super().__init__(
            embedding_dim=embedding_dim,
            normalize_embeddings=normalize_embeddings,
            projection_dim=projection_dim,
            feature_dims=feature_dims,
            num_norm_groups=num_norm_groups,
            num_res_blocks=num_res_blocks,
            save_timesteps=save_timesteps or [],
            gating_tempature=gating_tempature,
        )


class MoEFuserEmbedder(EmbeddingProjectorMixin, MoEWeightedFuser):
    """Mixture-of-Experts fuser with an embedding projection head."""

    def __init__(
        self,
        feature_dims,
        projection_dim: int = 384,
        embedding_dim: int = 256,
        normalize_embeddings: bool = True,
        num_norm_groups: int = 32,
        num_res_blocks: int = 1,
        save_timesteps=None,
        num_classes=None,
        num_experts: int = 8,
        top_k: int = 2,
    ):
        super().__init__(
            embedding_dim=embedding_dim,
            normalize_embeddings=normalize_embeddings,
            projection_dim=projection_dim,
            feature_dims=feature_dims,
            num_norm_groups=num_norm_groups,
            num_res_blocks=num_res_blocks,
            save_timesteps=save_timesteps or [],
            num_experts=num_experts,
            top_k=top_k,
        )
