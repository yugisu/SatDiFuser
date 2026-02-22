import torch
import torch.nn as nn
from archs.classifiers import GWFuserClassifier, LWFuserClassifier, MoEFuserClassifier
from archs.segmenters import GWFuserSegUPerNet, LWFuserSegUPerNet, MoEFuserSegUPerNet
from archs.embedders import GWFuserEmbedder, LWFuserEmbedder, MoEFuserEmbedder


class VICRegLoss(nn.Module):
    """
    VICReg adapted for geographic retrieval.

    Reference: Bardes et al., "VICReg: Variance-Invariance-Covariance
    Regularization for Self-Supervised Learning", ICLR 2022.
    https://arxiv.org/abs/2105.04906

    The three terms:

    * **Invariance** — MSE between embeddings of geographically adjacent
      patches (centroids within ``geo_positive_threshold_m`` metres).
      Pulls nearby patches together in embedding space.
    * **Variance** — hinge loss that keeps the per-dimension standard
      deviation of the batch embeddings above ``gamma``.  Prevents collapse
      to a constant vector.
    * **Covariance** — penalises off-diagonal entries of the batch
      covariance matrix.  Encourages each dimension to carry independent
      information.

    Loss = lambda_inv * L_inv  +  mu_var * L_var  +  nu_cov * L_cov

    ``forward(embeddings, coords)`` where ``coords`` is ``(B, 2)``
    float32 tensor of ``[lat, lon]`` in degrees (``NaN`` for missing).
    """

    def __init__(
        self,
        lambda_inv: float = 25.0,
        mu_var: float = 25.0,
        nu_cov: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
        geo_positive_threshold_m: float = 1000.0,
    ):
        super().__init__()
        self.lambda_inv = lambda_inv
        self.mu_var = mu_var
        self.nu_cov = nu_cov
        self.gamma = gamma
        self.eps = eps
        self.geo_positive_threshold_m = geo_positive_threshold_m

    @staticmethod
    def _haversine_km(lats: torch.Tensor, lons: torch.Tensor) -> torch.Tensor:
        """Full pairwise Haversine distance matrix in kilometres."""
        R = 6371.0
        lat_r = torch.deg2rad(lats)
        lon_r = torch.deg2rad(lons)
        dlat = lat_r.unsqueeze(0) - lat_r.unsqueeze(1)
        dlon = lon_r.unsqueeze(0) - lon_r.unsqueeze(1)
        a = (
            torch.sin(dlat / 2) ** 2
            + torch.cos(lat_r.unsqueeze(0))
            * torch.cos(lat_r.unsqueeze(1))
            * torch.sin(dlon / 2) ** 2
        )
        return 2.0 * R * torch.asin(a.clamp(0.0, 1.0).sqrt())

    def forward(self, embeddings: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: ``(B, D)`` — L2-normalised embedding vectors.
            coords:     ``(B, 2)`` float32 — ``[lat, lon]`` in degrees;
                        use ``NaN`` for samples without coordinates.
        Returns:
            Scalar loss.
        """
        B, D = embeddings.shape

        # ── Variance term ──────────────────────────────────────────────────
        # Centre embeddings across the batch, then penalise low per-dimension
        # standard deviation.  Note: embeddings are already L2-normalised
        # globally, but we still want batch-level spread per dimension.
        z = embeddings - embeddings.mean(dim=0, keepdim=True)  # (B, D)
        std = (z.var(dim=0) + self.eps).sqrt()  # (D,)
        var_loss = torch.clamp(self.gamma - std, min=0.0).mean()

        # ── Covariance term ────────────────────────────────────────────────
        # Off-diagonal entries of the normalised covariance matrix should be 0.
        cov = (z.T @ z) / max(B - 1, 1)  # (D, D)
        off_diag = cov.pow(2)
        off_diag.fill_diagonal_(0.0)
        cov_loss = off_diag.sum() / D

        # ── Invariance term (geographic) ────────────────────────────────────
        # Pull together embeddings of patches whose WGS84 centroids are within
        # geo_positive_threshold_m metres of each other.
        valid = ~torch.isnan(coords[:, 0]) & ~torch.isnan(coords[:, 1])  # (B,)
        if valid.sum() >= 2:
            dist_km = self._haversine_km(
                coords[:, 0].float(), coords[:, 1].float()
            )  # (B, B)
            threshold_km = self.geo_positive_threshold_m / 1000.0
            pos_mask = (
                (dist_km <= threshold_km) & valid.unsqueeze(0) & valid.unsqueeze(1)
            )
            pos_mask.fill_diagonal_(False)

            if pos_mask.any():
                diff = embeddings.unsqueeze(0) - embeddings.unsqueeze(1)  # (B, B, D)
                mse_pairs = diff.pow(2).sum(dim=-1)  # (B, B)
                inv_loss = mse_pairs[pos_mask].mean()
            else:
                inv_loss = embeddings.sum() * 0.0
        else:
            inv_loss = embeddings.sum() * 0.0

        return (
            self.lambda_inv * inv_loss + self.mu_var * var_loss + self.nu_cov * cov_loss
        )


def create_decoder(cfg, extraction_dims):
    base_task = cfg.task
    variant = cfg.fuser

    common_kwargs = {
        "feature_dims": extraction_dims,
        "projection_dim": cfg.projection_dim,
        "save_timesteps": cfg.save_timesteps,
        "num_classes": cfg.num_classes,
    }

    if base_task in ["classification", "multi_label_classification"]:
        cls_map = {
            "gw": (GWFuserClassifier, {}),
            "lw": (LWFuserClassifier, {}),
            "moe": (
                MoEFuserClassifier,
                {
                    "num_experts": cfg.get("num_experts"),
                    "top_k": cfg.get("top_k"),
                },
            ),
        }
    elif base_task in ["segmentation"]:
        common_kwargs.update(
            {
                "pool_scales": cfg.get("pool_scales"),
                "rescales": cfg.get("rescales"),
                "channels": cfg.get("channels"),
            }
        )
        cls_map = {
            "gw": (GWFuserSegUPerNet, {}),
            "lw": (LWFuserSegUPerNet, {}),
            "moe": (
                MoEFuserSegUPerNet,
                {
                    "num_experts": cfg.get("num_experts"),
                    "top_k": cfg.get("top_k"),
                },
            ),
        }

    elif base_task in ["embedding"]:
        common_kwargs.update(
            {
                "embedding_dim": cfg.get("embedding_dim", 256),
                "normalize_embeddings": cfg.get("normalize_embeddings", True),
            }
        )
        cls_map = {
            "gw": (GWFuserEmbedder, {}),
            "lw": (LWFuserEmbedder, {}),
            "moe": (
                MoEFuserEmbedder,
                {
                    "num_experts": cfg.get("num_experts"),
                    "top_k": cfg.get("top_k"),
                },
            ),
        }

    else:
        raise ValueError(f"Unknown base task: {base_task}")

    if variant not in cls_map:
        raise ValueError(f"Unknown variant '{variant}' for task '{base_task}'")

    decoder, extra_cfg_attrs = cls_map[variant]
    kwargs = common_kwargs.copy()
    for kwarg, cfg_attr in extra_cfg_attrs.items():
        kwargs[kwarg] = cfg_attr

    return decoder(**kwargs)


def set_criterion(task_type, ignore_index=None, vicreg_kwargs=None):

    if task_type in ["segmentation", "classification"]:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    elif task_type in ["multi_label_classification"]:
        criterion = torch.nn.BCEWithLogitsLoss()
    elif task_type in ["embedding"]:
        criterion = VICRegLoss(**(vicreg_kwargs or {}))
    else:
        raise ValueError(f"Task type {task_type} not supported")

    return criterion
