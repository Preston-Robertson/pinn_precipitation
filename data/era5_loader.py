"""
ERA5 Data Loader
================
Downloads and preprocesses ERA5 reanalysis data for PINN training.

Required ERA5 variables:
    - total_precipitation       (P)  [m]     → convert to mm/hr
    - specific_humidity         (q)  [kg/kg]
    - 10m_u_component_of_wind   (u)  [m/s]
    - 10m_v_component_of_wind   (v)  [m/s]
    - evaporation               (E)  [m]     → convert to mm/hr

Setup:
    pip install cdsapi xarray netCDF4 numpy
    Configure ~/.cdsapirc with your CDS API key:
        url: https://cds.climate.copernicus.eu/api/v2
        key: <UID>:<API-KEY>

Usage:
    downloader = ERA5Downloader(region=[60, -10, 35, 40])  # Europe
    downloader.download("era5_sample.nc", year=2020, months=[6,7,8])

    dataset = ERA5Dataset("era5_sample.nc")
    loader  = DataLoader(dataset, batch_size=512, shuffle=True)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional
import logging

log = logging.getLogger(__name__)


# ── Downloading ───────────────────────────────────────────────────────────────

class ERA5Downloader:
    """Thin wrapper around cdsapi for downloading the variables we need."""

    VARIABLES = [
        "total_precipitation",
        "specific_humidity",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "evaporation",
    ]

    def __init__(self, region: List[float] = None, pressure_level: bool = False):
        """
        Args:
            region: [north, west, south, east] in degrees.
                    Defaults to global (90, -180, -90, 180).
        """
        self.region = region or [90, -180, -90, 180]

    def download(
        self,
        output_path: str,
        year: int = 2020,
        months: List[int] = None,
        hours: List[str] = None,
    ):
        """Download ERA5 data to a NetCDF file."""
        try:
            import cdsapi
        except ImportError:
            raise ImportError("Run: pip install cdsapi")

        months = months or list(range(1, 13))
        hours = hours or ["00:00", "06:00", "12:00", "18:00"]

        c = cdsapi.Client()
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": self.VARIABLES,
                "year": str(year),
                "month": [f"{m:02d}" for m in months],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": hours,
                "area": self.region,
                "format": "netcdf",
            },
            output_path,
        )
        log.info(f"ERA5 data saved to {output_path}")


# ── Preprocessing ─────────────────────────────────────────────────────────────

class ERA5Normalizer:
    """Stores mean/std for each feature and normalises tensors."""

    def __init__(self):
        self.stats: dict = {}

    def fit(self, data: dict):
        """Compute normalisation stats from a dict of numpy arrays."""
        for name, arr in data.items():
            self.stats[name] = {
                "mean": float(arr.mean()),
                "std":  float(arr.std()) + 1e-8,
            }

    def transform(self, name: str, arr: np.ndarray) -> np.ndarray:
        s = self.stats[name]
        return (arr - s["mean"]) / s["std"]

    def inverse_transform_precip(self, arr: np.ndarray) -> np.ndarray:
        s = self.stats["P"]
        return arr * s["std"] + s["mean"]

    def save(self, path: str):
        import json
        with open(path, "w") as f:
            json.dump(self.stats, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ERA5Normalizer":
        import json
        obj = cls()
        with open(path) as f:
            obj.stats = json.load(f)
        return obj


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class ERA5Dataset(Dataset):
    """
    Loads ERA5 NetCDF data and exposes flattened (point, time) samples.

    Each sample is:
        coords: [lon_norm, lat_norm, t_norm]       shape (3,)
        atm:    [q_norm, u_norm, v_norm, E_norm]   shape (4,)
        P:      [P_norm]                           shape (1,)

    Finite-difference humidity derivatives (dq/dt, dq/dx, dq/dy) are also
    pre-computed and returned for efficient physics loss evaluation.
    """

    def __init__(self, nc_path: str, normalizer: Optional[ERA5Normalizer] = None):
        try:
            import xarray as xr
        except ImportError:
            raise ImportError("Run: pip install xarray netCDF4")

        ds = xr.open_dataset(nc_path)

        # ── Extract arrays (T, lat, lon) ──────────────────────────────────
        P_raw = ds["tp"].values * 1000.0      # m → mm  (hourly accumulation)
        q_raw = ds["q"].values
        u_raw = ds["u10"].values
        v_raw = ds["v10"].values
        E_raw = np.abs(ds["e"].values) * 1000  # ERA5 evap is negative, m → mm

        lats = ds["latitude"].values
        lons = ds["longitude"].values
        times = np.arange(P_raw.shape[0], dtype=np.float32)

        # ── Normalise ─────────────────────────────────────────────────────
        if normalizer is None:
            normalizer = ERA5Normalizer()
            normalizer.fit({"P": P_raw, "q": q_raw, "u": u_raw, "v": v_raw, "E": E_raw})
        self.normalizer = normalizer

        P = normalizer.transform("P", P_raw)
        q = normalizer.transform("q", q_raw)
        u = normalizer.transform("u", u_raw)
        v = normalizer.transform("v", v_raw)
        E = normalizer.transform("E", E_raw)

        # Normalise coordinates to [-1, 1]
        lon_n = (lons - lons.mean()) / (lons.std() + 1e-8)
        lat_n = (lats - lats.mean()) / (lats.std() + 1e-8)
        t_n   = (times - times.mean()) / (times.std() + 1e-8)

        # ── Finite-difference humidity gradients ──────────────────────────
        dq_dt = np.gradient(q, axis=0)   # ∂q/∂t
        dq_dy = np.gradient(q, axis=1)   # ∂q/∂lat  (y direction)
        dq_dx = np.gradient(q, axis=2)   # ∂q/∂lon  (x direction)

        # ── Flatten to (N_samples, ...) ───────────────────────────────────
        T, H, W = P.shape
        LON, LAT = np.meshgrid(lon_n, lat_n)  # (H, W) each

        # Broadcast coords across time
        LON_T = np.broadcast_to(LON[None], (T, H, W)).reshape(-1)
        LAT_T = np.broadcast_to(LAT[None], (T, H, W)).reshape(-1)
        TIM_T = np.broadcast_to(t_n[:, None, None], (T, H, W)).reshape(-1)

        self.coords = torch.tensor(
            np.stack([LON_T, LAT_T, TIM_T], axis=-1), dtype=torch.float32
        )
        self.atm = torch.tensor(
            np.stack([
                q.reshape(-1),
                u.reshape(-1),
                v.reshape(-1),
                E.reshape(-1),
            ], axis=-1),
            dtype=torch.float32,
        )
        self.P = torch.tensor(P.reshape(-1, 1), dtype=torch.float32)
        self.dq_dt = torch.tensor(dq_dt.reshape(-1, 1), dtype=torch.float32)
        self.dq_dx = torch.tensor(dq_dx.reshape(-1, 1), dtype=torch.float32)
        self.dq_dy = torch.tensor(dq_dy.reshape(-1, 1), dtype=torch.float32)

        log.info(f"ERA5Dataset: {len(self)} samples loaded from {nc_path}")

    def __len__(self) -> int:
        return len(self.P)

    def __getitem__(self, idx) -> dict:
        return {
            "coords": self.coords[idx],   # (3,)
            "atm":    self.atm[idx],      # (4,)
            "P":      self.P[idx],        # (1,)
            "dq_dt":  self.dq_dt[idx],    # (1,)
            "dq_dx":  self.dq_dx[idx],    # (1,)
            "dq_dy":  self.dq_dy[idx],    # (1,)
        }


def make_dataloader(
    nc_path: str,
    batch_size: int = 512,
    val_split: float = 0.1,
    num_workers: int = 4,
    normalizer: Optional[ERA5Normalizer] = None,
) -> Tuple[DataLoader, DataLoader, ERA5Normalizer]:
    """
    Build train/val DataLoaders from an ERA5 NetCDF file.

    Returns:
        train_loader, val_loader, normalizer
    """
    dataset = ERA5Dataset(nc_path, normalizer=normalizer)

    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, dataset.normalizer
