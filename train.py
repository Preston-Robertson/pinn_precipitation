"""
Training Loop
=============
Trains the precipitation PINN with curriculum learning:
  - Phase 1 (epochs 1–N/2):  λ_physics starts low (0.001), data-driven warm-up
  - Phase 2 (epochs N/2–N):  λ_physics anneals up to 1.0, physics increasingly enforced

Run:
    python train.py --config configs/default.yaml
    python train.py --data era5_sample.nc --epochs 100 --batch_size 512
"""

import argparse
import logging
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.pinn import PrecipitationPINN
from models.losses import PINNLoss, water_vapor_residual
from data.era5_loader import make_dataloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Training step ─────────────────────────────────────────────────────────────

def train_step(model, batch, criterion, optimizer, device):
    model.train()
    optimizer.zero_grad()

    coords = batch["coords"].to(device)          # (B, 3)
    atm    = batch["atm"].to(device)             # (B, 4)
    P_true = batch["P"].to(device)               # (B, 1)
    dq_dt  = batch["dq_dt"].to(device)           # (B, 1)
    dq_dx  = batch["dq_dx"].to(device)           # (B, 1)
    dq_dy  = batch["dq_dy"].to(device)           # (B, 1)

    # Atmospheric variables for physics residual
    q = atm[:, 0:1]
    u = atm[:, 1:2]
    v = atm[:, 2:3]
    E = atm[:, 3:4]

    # Forward pass
    P_pred = model(coords, atm)                  # (B, 1)

    # Physics residual using pre-computed finite-difference derivatives
    residual = water_vapor_residual(
        P=P_pred, q=q, u=u, v=v, E=E,
        coords=coords,
        dq_dt_fd=dq_dt,
        dq_dx_fd=dq_dx,
        dq_dy_fd=dq_dy,
    )

    losses = criterion(P_pred, P_true, residual)
    losses["total"].backward()

    # Gradient clipping — important for PINNs (autograd can spike)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {k: v.item() for k, v in losses.items()}


@torch.no_grad()
def val_step(model, batch, criterion, device):
    model.eval()

    coords = batch["coords"].to(device)
    atm    = batch["atm"].to(device)
    P_true = batch["P"].to(device)
    dq_dt  = batch["dq_dt"].to(device)
    dq_dx  = batch["dq_dx"].to(device)
    dq_dy  = batch["dq_dy"].to(device)

    q = atm[:, 0:1]
    u = atm[:, 1:2]
    v = atm[:, 2:3]
    E = atm[:, 3:4]

    P_pred = model(coords, atm)
    residual = water_vapor_residual(
        P=P_pred, q=q, u=u, v=v, E=E,
        coords=coords,
        dq_dt_fd=dq_dt,
        dq_dx_fd=dq_dx,
        dq_dy_fd=dq_dy,
    )

    losses = criterion(P_pred, P_true, residual)
    return {k: v.item() for k, v in losses.items()}


# ── Main training loop ────────────────────────────────────────────────────────

def train(
    data_path: str,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 3e-4,
    hidden_width: int = 256,
    n_residual_blocks: int = 4,
    lambda_data: float = 1.0,
    lambda_physics_start: float = 0.001,
    lambda_physics_max: float = 1.0,
    checkpoint_dir: str = "checkpoints",
    device: str = "auto",
):
    # ── Device ────────────────────────────────────────────────────────────
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    log.info(f"Loading ERA5 data from {data_path} ...")
    train_loader, val_loader, normalizer = make_dataloader(
        data_path, batch_size=batch_size
    )
    normalizer.save(f"{checkpoint_dir}/normalizer.json")
    log.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = PrecipitationPINN(
        hidden_width=hidden_width,
        n_residual_blocks=n_residual_blocks,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters: {n_params:,}")

    # ── Optimiser + scheduler ─────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # ── Loss with curriculum ──────────────────────────────────────────────
    criterion = PINNLoss(
        lambda_data=lambda_data,
        lambda_physics=lambda_physics_start,
    )

    # Anneal physics weight across the second half of training
    anneal_start_epoch = epochs // 2
    anneal_factor = (lambda_physics_max / lambda_physics_start) ** (
        1.0 / max(epochs - anneal_start_epoch, 1)
    )

    # ── Training ──────────────────────────────────────────────────────────
    Path(checkpoint_dir).mkdir(exist_ok=True)
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Curriculum: start annealing physics weight in second half
        if epoch >= anneal_start_epoch:
            criterion.anneal_physics_weight(
                factor=anneal_factor,
                max_weight=lambda_physics_max,
            )

        # Train
        train_losses = {"total": 0.0, "data": 0.0, "physics": 0.0}
        for batch in train_loader:
            step_losses = train_step(model, batch, criterion, optimizer, device)
            for k in train_losses:
                train_losses[k] += step_losses[k]
        for k in train_losses:
            train_losses[k] /= len(train_loader)

        # Validate
        val_losses = {"total": 0.0, "data": 0.0, "physics": 0.0}
        for batch in val_loader:
            step_losses = val_step(model, batch, criterion, device)
            for k in val_losses:
                val_losses[k] += step_losses[k]
        for k in val_losses:
            val_losses[k] /= len(val_loader)

        scheduler.step()

        elapsed = time.time() - t0
        log.info(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train total={train_losses['total']:.4f} data={train_losses['data']:.4f} phys={train_losses['physics']:.4f} | "
            f"Val total={val_losses['total']:.4f} data={val_losses['data']:.4f} phys={val_losses['physics']:.4f} | "
            f"λ_phys={criterion.lambda_physics:.4f} | {elapsed:.1f}s"
        )

        history.append({"epoch": epoch, "train": train_losses, "val": val_losses,
                        "lambda_physics": criterion.lambda_physics})

        # Checkpoint
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "hparams": {
                        "hidden_width": hidden_width,
                        "n_residual_blocks": n_residual_blocks,
                    },
                },
                f"{checkpoint_dir}/best_model.pt",
            )
            log.info(f"  ✓ New best model saved (val_loss={best_val_loss:.4f})")

    log.info("Training complete.")
    return model, history


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Precipitation PINN")
    parser.add_argument("--data",   required=True,     help="Path to ERA5 NetCDF file")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr",     type=float, default=3e-4)
    parser.add_argument("--hidden_width", type=int, default=256)
    parser.add_argument("--n_blocks",     type=int, default=4)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    train(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_width=args.hidden_width,
        n_residual_blocks=args.n_blocks,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )
