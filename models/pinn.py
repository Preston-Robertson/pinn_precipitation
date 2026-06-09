"""
Physics-Informed Neural Network for Precipitation Prediction
============================================================
Physics basis: Atmospheric Water Vapor Continuity Equation

    ∂q/∂t + u·∂q/∂x + v·∂q/∂y = E - P

Where:
    q   = specific humidity  [kg/kg]
    u,v = horizontal wind components [m/s]
    E   = evaporation rate   [kg/m²/s]
    P   = precipitation rate [kg/m²/s]  ← what we predict

The PINN learns P(x, y, t, features) such that:
  1. Data loss:    predicted P ≈ ERA5 observed P
  2. Physics loss: residual of the continuity equation ≈ 0
"""

import torch
import torch.nn as nn
from typing import Tuple


class FourierEmbedding(nn.Module):
    """
    Random Fourier Feature embedding for spatiotemporal inputs.
    Helps the network capture multi-scale periodic patterns
    (diurnal cycles, seasonal variation, spatial wavelengths).
    """

    def __init__(self, input_dim: int, num_features: int = 64, scale: float = 1.0):
        super().__init__()
        # Fixed random frequencies — not trained
        B = torch.randn(input_dim, num_features) * scale
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim)
        proj = x @ self.B  # (batch, num_features)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (batch, 2*num_features)

    @property
    def output_dim(self) -> int:
        return self.B.shape[1] * 2


class ResidualBlock(nn.Module):
    """Single residual block with layer norm — stabilises deep PINN training."""

    def __init__(self, width: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class PrecipitationPINN(nn.Module):
    """
    Physics-Informed Neural Network for precipitation prediction.

    Input features (per grid point per time step):
        Spatiotemporal:  x (lon), y (lat), t (time)
        Atmospheric:     q (specific humidity), u (u-wind), v (v-wind), E (evaporation)

    Outputs:
        P  — precipitation rate [mm/hr, later de-normalised]

    Architecture:
        Fourier embedding → MLP with residual blocks → softplus output
        (softplus ensures P ≥ 0 physically)
    """

    def __init__(
        self,
        n_atm_features: int = 4,   # q, u, v, E
        fourier_features: int = 64,
        fourier_scale: float = 2.0,
        hidden_width: int = 256,
        n_residual_blocks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ── Spatiotemporal embedding (x, y, t) ──────────────────────────────
        self.fourier = FourierEmbedding(
            input_dim=3,
            num_features=fourier_features,
            scale=fourier_scale,
        )
        embed_dim = self.fourier.output_dim  # 2 * fourier_features

        # ── Input projection: concat(embedding, atm_features) → hidden_width
        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim + n_atm_features, hidden_width),
            nn.LayerNorm(hidden_width),
            nn.Tanh(),
        )

        # ── Residual trunk ───────────────────────────────────────────────────
        self.trunk = nn.Sequential(
            *[ResidualBlock(hidden_width, dropout) for _ in range(n_residual_blocks)]
        )

        # ── Output head → scalar precipitation ──────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(hidden_width, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softplus(),  # P ≥ 0
        )

    def forward(
        self,
        coords: torch.Tensor,   # (B, 3)  — [lon, lat, time]  normalised to [-1,1]
        atm: torch.Tensor,      # (B, 4)  — [q, u, v, E]      normalised
    ) -> torch.Tensor:
        """Returns predicted precipitation (B, 1)."""
        emb = self.fourier(coords)              # (B, embed_dim)
        x = torch.cat([emb, atm], dim=-1)      # (B, embed_dim + n_atm)
        x = self.input_proj(x)
        x = self.trunk(x)
        return self.head(x)                     # (B, 1)

    def forward_with_grads(
        self,
        coords: torch.Tensor,
        atm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns ∂P/∂t, ∂q/∂x, ∂q/∂y via autograd.
        Used inside the physics loss computation.
        """
        coords = coords.requires_grad_(True)
        P = self.forward(coords, atm)

        # Gradients of P w.r.t. coordinates
        grads = torch.autograd.grad(
            P, coords,
            grad_outputs=torch.ones_like(P),
            create_graph=True,
        )[0]  # (B, 3): [∂P/∂x, ∂P/∂y, ∂P/∂t]

        dP_dx = grads[:, 0:1]
        dP_dy = grads[:, 1:2]
        dP_dt = grads[:, 2:3]

        return P, dP_dx, dP_dy, dP_dt
