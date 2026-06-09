"""
Inference & Evaluation
======================
Load a trained checkpoint and evaluate on held-out ERA5 data.

Metrics:
    - RMSE, MAE, Bias
    - Pearson correlation
    - Fraction Skill Score (FSS) for spatial verification
    - Physics residual magnitude

Usage:
    from utils.evaluate import evaluate_checkpoint
    results = evaluate_checkpoint("checkpoints/best_model.pt", "era5_test.nc")
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional
import logging

log = logging.getLogger(__name__)


def load_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
):
    """Load model from checkpoint."""
    from models.pinn import PrecipitationPINN

    ckpt = torch.load(checkpoint_path, map_location=device)
    hparams = ckpt.get("hparams", {})

    model = PrecipitationPINN(
        hidden_width=hparams.get("hidden_width", 256),
        n_residual_blocks=hparams.get("n_residual_blocks", 4),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    log.info(f"Loaded model from epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")
    return model


@torch.no_grad()
def predict(
    model,
    coords: torch.Tensor,   # (N, 3)
    atm: torch.Tensor,      # (N, 4)
    batch_size: int = 4096,
    device: str = "cpu",
) -> np.ndarray:
    """Run inference in batches. Returns predicted P (N,)."""
    model.eval()
    preds = []
    for i in range(0, len(coords), batch_size):
        c = coords[i : i + batch_size].to(device)
        a = atm[i : i + batch_size].to(device)
        p = model(c, a)
        preds.append(p.cpu().numpy())
    return np.concatenate(preds, axis=0).squeeze()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Standard regression metrics for precipitation evaluation."""
    from scipy.stats import pearsonr

    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    bias = np.mean(y_pred - y_true)
    corr, _ = pearsonr(y_true.ravel(), y_pred.ravel())

    return {
        "MAE":  float(mae),
        "RMSE": float(rmse),
        "Bias": float(bias),
        "Corr": float(corr),
    }


def evaluate_checkpoint(
    checkpoint_path: str,
    nc_path: str,
    normalizer_path: Optional[str] = None,
    device: str = "cpu",
) -> dict:
    """Full evaluation pipeline."""
    from data.era5_loader import ERA5Dataset, ERA5Normalizer
    from models.losses import water_vapor_residual

    normalizer = None
    if normalizer_path:
        normalizer = ERA5Normalizer.load(normalizer_path)

    model = load_checkpoint(checkpoint_path, device=device)
    dataset = ERA5Dataset(nc_path, normalizer=normalizer)

    P_pred = predict(model, dataset.coords, dataset.atm, device=device)

    # De-normalise
    P_true_norm = dataset.P.numpy().squeeze()
    P_true_phys = normalizer.inverse_transform_precip(P_true_norm) if normalizer else P_true_norm
    P_pred_phys = normalizer.inverse_transform_precip(P_pred) if normalizer else P_pred

    metrics = compute_metrics(P_true_phys, P_pred_phys)

    # Physics residual on a sample
    sample_size = min(10_000, len(dataset))
    idx = np.random.choice(len(dataset), sample_size, replace=False)
    coords_s = dataset.coords[idx]
    atm_s    = dataset.atm[idx]
    P_pred_t = torch.tensor(P_pred[idx, None])
    dq_dt_s  = dataset.dq_dt[idx]
    dq_dx_s  = dataset.dq_dx[idx]
    dq_dy_s  = dataset.dq_dy[idx]

    residual = water_vapor_residual(
        P=P_pred_t,
        q=atm_s[:, 0:1],
        u=atm_s[:, 1:2],
        v=atm_s[:, 2:3],
        E=atm_s[:, 3:4],
        coords=coords_s,
        dq_dt_fd=dq_dt_s,
        dq_dx_fd=dq_dx_s,
        dq_dy_fd=dq_dy_s,
    )
    metrics["PhysicsResidualRMSE"] = float(residual.pow(2).mean().sqrt())

    log.info("Evaluation results:")
    for k, v in metrics.items():
        log.info(f"  {k}: {v:.4f}")

    return metrics
