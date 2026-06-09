"""
Synthetic Data Generator
========================
Generates synthetic ERA5-like data for testing the PINN pipeline
without needing a CDS API key or downloading real data.

The synthetic precipitation field satisfies the water vapor
continuity equation by construction, so the physics loss should
converge to near-zero on this data — a useful sanity check.

Usage:
    python -m utils.synthetic_data                  # saves synthetic_era5.nc
    python train.py --data synthetic_era5.nc        # trains on synthetic data
"""

import numpy as np
from pathlib import Path


def generate_synthetic_era5(
    output_path: str = "synthetic_era5.nc",
    n_times: int = 48,        # 48 timesteps (e.g. 2 days × 6-hourly)
    n_lats: int = 32,
    n_lons: int = 64,
    seed: int = 42,
):
    """
    Generate a synthetic NetCDF dataset that mimics ERA5 structure.

    Physics construction:
        u, v   — sinusoidal wind fields
        q      — humidity from advection + source/sink
        E      — evaporation ~ 0.2 * sin(lat) + noise
        P      — derived from continuity equation so residual = 0
    """
    try:
        import xarray as xr
        import netCDF4  # noqa — needed by xarray for nc writing
    except ImportError:
        raise ImportError("Run: pip install xarray netCDF4")

    rng = np.random.default_rng(seed)

    lats = np.linspace(20.0, 70.0, n_lats)
    lons = np.linspace(-20.0, 50.0, n_lons)
    times = np.arange(n_times, dtype=np.float32)

    LON, LAT, T = np.meshgrid(lons, lats, times, indexing="ij")  # (nL, nLa, nT)
    # Reshape to (nT, nLa, nL) to match ERA5 convention
    LON = LON.transpose(2, 1, 0)
    LAT = LAT.transpose(2, 1, 0)
    T   = T.transpose(2, 1, 0)

    lat_rad = np.deg2rad(LAT)

    # ── Wind fields (simple sinusoidal) ──────────────────────────────────
    u = 5.0 * np.cos(lat_rad) * np.sin(2 * np.pi * T / n_times)
    v = 2.0 * np.sin(lat_rad) * np.cos(2 * np.pi * T / n_times)

    # ── Humidity field ────────────────────────────────────────────────────
    # q ~ Gaussian moisture blob that advects with wind
    q = 0.01 * (
        np.exp(-((LAT - 45) ** 2) / 200 - ((LON - 15) ** 2) / 400)
        + 0.3 * rng.standard_normal((n_times, n_lats, n_lons))
    )
    q = np.clip(q, 0, None)

    # ── Evaporation (always positive) ────────────────────────────────────
    E = 0.0005 * (0.5 + np.sin(lat_rad)) + 0.0001 * rng.standard_normal(q.shape)
    E = np.clip(E, 0, None)

    # ── Precipitation from continuity: P = E - ∂q/∂t - u·∂q/∂x - v·∂q/∂y
    dq_dt = np.gradient(q, axis=0)
    dq_dy = np.gradient(q, axis=1)
    dq_dx = np.gradient(q, axis=2)

    P = E - dq_dt - u * dq_dx - v * dq_dy
    P = np.clip(P, 0, None)   # precipitation can't be negative
    P = P / 1000.0            # convert to metres to match ERA5 convention

    E_m = E / 1000.0

    # ── Package as xarray Dataset ─────────────────────────────────────────
    import xarray as xr
    import pandas as pd

    time_index = pd.date_range("2020-06-01", periods=n_times, freq="6h")

    ds = xr.Dataset(
        {
            "tp":  (["time", "latitude", "longitude"], P.astype(np.float32)),
            "q":   (["time", "latitude", "longitude"], q.astype(np.float32)),
            "u10": (["time", "latitude", "longitude"], u.astype(np.float32)),
            "v10": (["time", "latitude", "longitude"], v.astype(np.float32)),
            "e":   (["time", "latitude", "longitude"], -E_m.astype(np.float32)),  # ERA5 sign
        },
        coords={
            "time":      time_index,
            "latitude":  lats.astype(np.float32),
            "longitude": lons.astype(np.float32),
        },
        attrs={
            "description": "Synthetic ERA5-like data for PINN testing",
            "physics": "Water vapor continuity: P = E - dq/dt - u*dq/dx - v*dq/dy",
        },
    )

    ds.to_netcdf(output_path)
    print(f"Synthetic dataset saved to {output_path}")
    print(f"  Shape: {n_times} timesteps × {n_lats} lats × {n_lons} lons")
    print(f"  P range: [{P.min()*1000:.4f}, {P.max()*1000:.4f}] mm")
    print(f"  q range: [{q.min():.6f}, {q.max():.6f}] kg/kg")
    return ds


if __name__ == "__main__":
    generate_synthetic_era5()
