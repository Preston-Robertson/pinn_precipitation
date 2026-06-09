# Precipitation PINN — PyTorch

A **Physics-Informed Neural Network** that predicts precipitation from ERA5 reanalysis data,
constrained by the atmospheric water vapor continuity equation.

---

## Physics

The governing PDE is the **water vapor continuity equation**:

```
∂q/∂t  +  u·∂q/∂x  +  v·∂q/∂y  =  E  -  P
```

| Symbol | Meaning              | ERA5 variable            |
|--------|----------------------|--------------------------|
| `q`    | Specific humidity    | `specific_humidity`      |
| `u`    | Zonal wind           | `10m_u_component_of_wind`|
| `v`    | Meridional wind      | `10m_v_component_of_wind`|
| `E`    | Evaporation          | `evaporation`            |
| `P`    | Precipitation        | `total_precipitation` ← **predicted** |

The PINN minimises:

```
L_total = λ_data · MSE(P_pred, P_ERA5) + λ_physics · MSE(residual, 0)
```

with curriculum learning: `λ_physics` starts near 0 and anneals up to 1.

---

## Project Structure

```
pinn_precipitation/
├── models/
│   ├── pinn.py          # Model architecture (Fourier embedding + residual MLP)
│   └── losses.py        # PINNLoss + physics residual computation
├── data/
│   └── era5_loader.py   # ERA5 download, normalisation, PyTorch Dataset
├── utils/
│   ├── synthetic_data.py  # Synthetic ERA5-like data for testing
│   └── evaluate.py        # Metrics and checkpoint evaluation
├── train.py             # Main training script
└── requirements.txt
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2a. Test with synthetic data (no ERA5 account needed)

```bash
# Generate synthetic data (~2 min on CPU)
python -m utils.synthetic_data

# Train
python train.py --data synthetic_era5.nc --epochs 50 --batch_size 256
```

The physics residual should converge to near-zero on synthetic data
(it satisfies the PDE by construction) — a useful sanity check.

### 2b. Download real ERA5 data

First, register at https://cds.climate.copernicus.eu and configure `~/.cdsapirc`:

```
url: https://cds.climate.copernicus.eu/api/v2
key: <UID>:<YOUR-API-KEY>
```

Then:

```python
from data.era5_loader import ERA5Downloader

dl = ERA5Downloader(region=[60, -10, 35, 40])   # Europe bounding box
dl.download("era5_europe_2020_jja.nc",
            year=2020, months=[6, 7, 8])         # JJA (summer)
```

```bash
python train.py \
  --data  era5_europe_2020_jja.nc \
  --epochs 100 \
  --batch_size 512 \
  --hidden_width 256 \
  --n_blocks 4
```

### 3. Evaluate

```python
from utils.evaluate import evaluate_checkpoint

metrics = evaluate_checkpoint(
    checkpoint_path="checkpoints/best_model.pt",
    nc_path="era5_test.nc",
    normalizer_path="checkpoints/normalizer.json",
)
# {'MAE': 0.12, 'RMSE': 0.19, 'Corr': 0.87, 'PhysicsResidualRMSE': 0.003, ...}
```

---

## Architecture

```
Input:  [lon, lat, t]  +  [q, u, v, E]
         ↓ Fourier embedding          ↓
         (sin/cos random features)    |
         ↓ concat ────────────────────┘
         Linear → LayerNorm → Tanh
         ↓
         4× Residual blocks (256-wide, LayerNorm, Tanh)
         ↓
         Linear(256→64) → Tanh → Linear(64→1) → Softplus
Output: P (precipitation, ≥ 0)
```

**Key design choices:**
- **Fourier embedding** captures multi-scale spatial/temporal patterns (diurnal, seasonal)
- **Softplus output** enforces P ≥ 0 without hard clipping gradients
- **LayerNorm** in residual blocks stabilises autograd-heavy PINN training
- **Curriculum learning** — physics weight anneals from 0.001 → 1.0 to prevent early training collapse

---

## Tips for Real ERA5 Data

| Challenge | Recommendation |
|-----------|---------------|
| Heavy-tailed P distribution | Log-transform: `log(P + 1e-5)` before normalising |
| Memory for large domains | Subsample spatially; use `dask`-backed xarray |
| Slow convergence | Increase `λ_physics` more slowly; use larger batch |
| NaN gradients | Check `clip_grad_norm_` (already set to 1.0) |
| Wet/dry imbalance | Weighted sampler or focal-style data loss |
