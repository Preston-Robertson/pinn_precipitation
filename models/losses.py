"""
Physics Loss Functions
======================
Implements the atmospheric water vapor continuity equation as a soft constraint.

Governing PDE:
    ∂q/∂t + u·∂q/∂x + v·∂q/∂y = E - P

Rearranging for the residual (what should be ≈ 0):
    residual = ∂q/∂t + u·∂q/∂x + v·∂q/∂y - E + P ≈ 0

We use autograd to compute ∂q/∂t, ∂q/∂x, ∂q/∂y from a humidity
surrogate (or from finite differences on the ERA5 grid), then
penalise the residual magnitude during training.
"""

import torch
import torch.nn as nn
from typing import Optional


def water_vapor_residual(
    P: torch.Tensor,             # (B,1)  predicted precipitation
    q: torch.Tensor,             # (B,1)  specific humidity
    u: torch.Tensor,             # (B,1)  u-wind
    v: torch.Tensor,             # (B,1)  v-wind
    E: torch.Tensor,             # (B,1)  evaporation
    coords: torch.Tensor,        # (B,3)  [lon, lat, t] — must have requires_grad=True
    dq_dt_fd: Optional[torch.Tensor] = None,  # finite-diff ∂q/∂t if available
    dq_dx_fd: Optional[torch.Tensor] = None,  # finite-diff ∂q/∂x if available
    dq_dy_fd: Optional[torch.Tensor] = None,  # finite-diff ∂q/∂y if available
) -> torch.Tensor:
    """
    Compute the PDE residual for the water vapour continuity equation.

    Strategy:
      - Prefer ERA5 finite-difference derivatives when available (more accurate).
      - Fall back to treating q as a function of coords via autograd (prototype mode).

    Returns residual tensor of shape (B, 1).
    """
    if dq_dt_fd is not None and dq_dx_fd is not None and dq_dy_fd is not None:
        # ── Mode 1: finite-difference derivatives from ERA5 grid ──────────
        dq_dt = dq_dt_fd
        dq_dx = dq_dx_fd
        dq_dy = dq_dy_fd
    else:
        # ── Mode 2: autograd (prototype / collocation points) ─────────────
        # q must have been computed from coords with a differentiable path.
        if not coords.requires_grad:
            raise ValueError(
                "coords must have requires_grad=True for autograd mode. "
                "Pass finite-difference derivatives or enable grad on coords."
            )
        q_sum = q.sum()
        grads = torch.autograd.grad(
            q_sum, coords, create_graph=True, retain_graph=True
        )[0]
        dq_dx = grads[:, 0:1]
        dq_dy = grads[:, 1:2]
        dq_dt = grads[:, 2:3]

    # ∂q/∂t + u·∂q/∂x + v·∂q/∂y = E - P
    # residual = ∂q/∂t + u·∂q/∂x + v·∂q/∂y - E + P
    residual = dq_dt + u * dq_dx + v * dq_dy - E + P
    return residual


class PINNLoss(nn.Module):
    """
    Combined loss for the precipitation PINN.

    L_total = λ_data · L_data + λ_phys · L_physics

    L_data  = MSE(P_pred, P_true)         — fit the observations
    L_phys  = MSE(residual, 0)            — satisfy the PDE

    The λ weights are tunable; a common strategy is to start with
    λ_phys = 0.01 and gradually increase it (curriculum learning).
    """

    def __init__(self, lambda_data: float = 1.0, lambda_physics: float = 0.01):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_physics = lambda_physics
        self.mse = nn.MSELoss()

    def forward(
        self,
        P_pred: torch.Tensor,
        P_true: torch.Tensor,
        physics_residual: torch.Tensor,
    ) -> dict:
        """
        Args:
            P_pred:           (B,1) model predictions
            P_true:           (B,1) ERA5 observed precipitation
            physics_residual: (B,1) water vapor continuity residual

        Returns dict with individual losses + weighted total.
        """
        L_data = self.mse(P_pred, P_true)
        L_phys = self.mse(physics_residual, torch.zeros_like(physics_residual))
        L_total = self.lambda_data * L_data + self.lambda_physics * L_phys

        return {
            "total": L_total,
            "data": L_data,
            "physics": L_phys,
        }

    def anneal_physics_weight(self, factor: float = 1.05, max_weight: float = 1.0):
        """Gradually increase physics weight (curriculum learning)."""
        self.lambda_physics = min(self.lambda_physics * factor, max_weight)
