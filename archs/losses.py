import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


# ---------------------------------------------------------------------------
# NN matching helpers (adapted from VICRegL / CAEVL)
# ---------------------------------------------------------------------------


def _batched_index_select(
    input: torch.Tensor, dim: int, index: torch.Tensor
) -> torch.Tensor:
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


def _nearest_neighbors(
    input_maps: torch.Tensor,
    candidate_maps: torch.Tensor,
    distances: torch.Tensor,
    num_matches: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select `num_matches` NN pairs ordered by ascending distance."""
    batch_size = input_maps.size(0)
    if num_matches is None or num_matches == -1:
        num_matches = input_maps.size(1)

    topk_values, topk_indices = distances.topk(k=1, largest=False)
    topk_values = topk_values.squeeze(-1)    # (B, HW)
    topk_indices = topk_indices.squeeze(-1)  # (B, HW)

    _, sorted_values_indices = torch.sort(topk_values, dim=1)
    sorted_indices, _ = torch.sort(sorted_values_indices, dim=1)

    mask = torch.stack(
        [
            torch.where(sorted_indices[i] < num_matches, True, False)
            for i in range(batch_size)
        ]
    )

    topk_indices_selected = topk_indices.masked_select(mask).reshape(batch_size, num_matches)
    indices = (
        torch.arange(0, topk_values.size(1))
        .unsqueeze(0)
        .repeat(batch_size, 1)
        .to(topk_values.device)
    )
    indices_selected = indices.masked_select(mask).reshape(batch_size, num_matches)

    filtered_input = _batched_index_select(input_maps, 1, indices_selected)
    filtered_candidate = _batched_index_select(candidate_maps, 1, topk_indices_selected)
    return filtered_input, filtered_candidate


def _nn_on_l2(
    input_maps: torch.Tensor,
    candidate_maps: torch.Tensor,
    num_matches: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest-neighbour matching in feature (L2) space.

    Args:
        input_maps:     (B, HW, C)
        candidate_maps: (B, HW, C)
    """
    distances = torch.cdist(input_maps, candidate_maps)
    return _nearest_neighbors(input_maps, candidate_maps, distances, num_matches)


def _nn_on_location(
    input_locations: torch.Tensor,
    candidate_locations: torch.Tensor,
    input_maps: torch.Tensor,
    candidate_maps: torch.Tensor,
    num_matches: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest-neighbour matching in geographic coordinate space.

    Args:
        input_locations:     (B, HW, 2)  coordinates for input patches
        candidate_locations: (B, HW, 2)  coordinates for candidate patches
        input_maps:          (B, HW, C)
        candidate_maps:      (B, HW, C)
    """
    distances = torch.cdist(input_locations.float(), candidate_locations.float())
    return _nearest_neighbors(input_maps, candidate_maps, distances, num_matches)


# ---------------------------------------------------------------------------
# SpatialLoss
# ---------------------------------------------------------------------------


class SpatialLoss(nn.Module):
    """VICReg / VICRegL-style loss for FuserEmbedder outputs.

    Combines:
      - **Global** VICReg loss on GeM-pooled embeddings.
      - **Local** VICRegL loss on spatial feature maps using mutual
        nearest-neighbour matching in feature space.

    ``alpha`` blends the two terms: ``alpha=1`` → global only, ``alpha=0`` → local only.

    Expected inputs come directly from ``FuserEmbedder.forward_unpooled``:
        spatial: (B, C, H, W)
        pooled:  (B, C)

    Args:
        alpha:            Weight of the global loss term (1 − alpha weights local).
        invariance_coeff: Coefficient for the MSE invariance term.
        std_coeff:        Coefficient for the variance regularisation term.
        cov_coeff:        Coefficient for the covariance regularisation term.
        num_matches:      NN pairs kept per sample in the local loss (None = all).
    """

    def __init__(
        self,
        alpha: float = 0.5,
        invariance_coeff: float = 25.0,
        std_coeff: float = 25.0,
        cov_coeff: float = 1.0,
        num_matches: int | None = None,
    ):
        super().__init__()

        self.alpha = alpha
        self.invariance_coeff = invariance_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.epsilon = 1e-5
        self.gamma = 1.0
        self.num_matches = num_matches

    # ------------------------------------------------------------------
    # Loss primitives
    # ------------------------------------------------------------------

    def _off_diagonal(self, x: torch.Tensor) -> torch.Tensor:
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def _vicreg_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """VICReg loss on global (pooled) embeddings. Returns scalar."""
        invariance_loss = F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + self.epsilon)
        std_y = torch.sqrt(y.var(dim=0) + self.epsilon)
        std_loss = (
            torch.mean(F.relu(self.gamma - std_x)) / 2
            + torch.mean(F.relu(self.gamma - std_y)) / 2
        )

        batch_size, num_features = x.shape
        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = self._off_diagonal(cov_x).pow_(2).sum().div(num_features)
        cov_loss += self._off_diagonal(cov_y).pow_(2).sum().div(num_features)

        return (
            self.invariance_coeff * invariance_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )

    def _vicregl_maps_loss(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """VICRegL component losses on NN-matched feature pairs.

        Args:
            x, y: (B, num_matches, C) matched feature pairs.

        Returns:
            Tuple of scalar (invariance_loss, std_loss, cov_loss).
        """
        invariance_loss = F.mse_loss(x, y)

        std_x = torch.sqrt(x.var(dim=0) + self.epsilon)
        std_y = torch.sqrt(y.var(dim=0) + self.epsilon)
        std_loss = (
            torch.mean(F.relu(self.gamma - std_x)) / 2
            + torch.mean(F.relu(self.gamma - std_y)) / 2
        )

        # Covariance computed per match position, averaged over positions.
        x_t = x.permute(1, 0, 2)  # (num_matches, B, C)
        y_t = y.permute(1, 0, 2)
        x_t = x_t - x_t.mean(dim=-2, keepdim=True)
        y_t = y_t - y_t.mean(dim=-2, keepdim=True)

        sample_size, num_channels = x_t.shape[-2], x_t.shape[-1]
        non_diag = ~torch.eye(num_channels, device=x.device, dtype=torch.bool)
        cov_x = torch.einsum("...bc,...bd->...cd", x_t, x_t) / (sample_size - 1)
        cov_y = torch.einsum("...bc,...bd->...cd", y_t, y_t) / (sample_size - 1)
        cov_loss = (
            cov_x[..., non_diag].pow(2).sum(-1) / num_channels
            + cov_y[..., non_diag].pow(2).sum(-1) / num_channels
        ).mean() / 2

        return (
            self.invariance_coeff * invariance_loss,
            self.std_coeff * std_loss,
            self.cov_coeff * cov_loss,
        )

    def _local_loss(
        self,
        spatial_1: torch.Tensor,
        spatial_2: torch.Tensor,
        locations_1: torch.Tensor | None = None,
        locations_2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mutual NN-matched VICRegL loss on spatial feature maps. Returns scalar.

        When ``locations_1`` and ``locations_2`` are provided (both required),
        an additional location-matched pass is run and averaged with the
        feature-matched pass.  Locations should be ``(B, H*W, 2)`` tensors of
        geographic or pixel coordinates for each patch in the respective view.
        """
        maps_1 = rearrange(spatial_1, "b c h w -> b (h w) c")
        maps_2 = rearrange(spatial_2, "b c h w -> b (h w) c")

        # Feature-based matching (always).
        m1_filt, m1_nn = _nn_on_l2(maps_1, maps_2, self.num_matches)
        m2_filt, m2_nn = _nn_on_l2(maps_2, maps_1, self.num_matches)

        inv1, var1, cov1 = self._vicregl_maps_loss(m1_filt, m1_nn)
        inv2, var2, cov2 = self._vicregl_maps_loss(m2_filt, m2_nn)

        feat_loss = (inv1 + inv2) / 2 + (var1 + var2) / 2 + (cov1 + cov2) / 2

        if locations_1 is None or locations_2 is None:
            return feat_loss

        # Location-based matching (when coordinates are available).
        lm1_filt, lm1_nn = _nn_on_location(locations_1, locations_2, maps_1, maps_2, self.num_matches)
        lm2_filt, lm2_nn = _nn_on_location(locations_2, locations_1, maps_2, maps_1, self.num_matches)

        linv1, lvar1, lcov1 = self._vicregl_maps_loss(lm1_filt, lm1_nn)
        linv2, lvar2, lcov2 = self._vicregl_maps_loss(lm2_filt, lm2_nn)

        loc_loss = (linv1 + linv2) / 2 + (lvar1 + lvar2) / 2 + (lcov1 + lcov2) / 2

        return (feat_loss + loc_loss) / 2

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        spatial_1: torch.Tensor,
        pooled_1: torch.Tensor,
        spatial_2: torch.Tensor,
        pooled_2: torch.Tensor,
        locations_1: torch.Tensor | None = None,
        locations_2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the combined VICReg + VICRegL loss.

        Args:
            spatial_1:   Spatial features from view 1  (B, C, H, W).
            pooled_1:    Pooled embedding from view 1  (B, C).
            spatial_2:   Spatial features from view 2  (B, C, H, W).
            pooled_2:    Pooled embedding from view 2  (B, C).
            locations_1: Optional patch coordinates for view 1  (B, H*W, 2).
            locations_2: Optional patch coordinates for view 2  (B, H*W, 2).

        Returns:
            Scalar loss tensor.
        """
        device = pooled_1.device

        if self.alpha > 0:
            global_loss = self._vicreg_loss(pooled_1, pooled_2)
        else:
            global_loss = torch.tensor(0.0, device=device)

        if self.alpha < 1:
            local_loss = self._local_loss(spatial_1, spatial_2, locations_1, locations_2)
        else:
            local_loss = torch.tensor(0.0, device=device)

        return self.alpha * global_loss + (1 - self.alpha) * local_loss
