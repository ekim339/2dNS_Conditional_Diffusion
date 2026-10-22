"""Evaluate the PIDM three-frame residual on ground-truth vorticity.

This script does not load or run a diffusion model. It reuses the exact
``DDPMTrainer.pde_residual`` implementation used during PIDM training and
reports global, element-weighted averages over physical-space data.

Run with the same array used for training:

    python PIDM/evaluate_ground_truth_physics.py \
        --data /content/drive/MyDrive/Lab/CondDiff/NSE_Data_new.npy

By default the script evaluates every valid five-frame center in the first
80 percent of the trajectory, matching the PIDM training split and center
eligibility. Only the center and its immediate neighbors are used by the
three-frame residual.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch

try:
    from .model import (
        DDPMTrainer,
        KOLMOGOROV_DT_PHYS,
        KOLMOGOROV_FORCING_WAVENUMBER,
        KOLMOGOROV_REYNOLDS_NUMBER,
        KOLMOGOROV_VISCOSITY,
        default_device,
    )
except ImportError:  # Direct execution: python PIDM/evaluate_ground_truth_physics.py
    from model import (
        DDPMTrainer,
        KOLMOGOROV_DT_PHYS,
        KOLMOGOROV_FORCING_WAVENUMBER,
        KOLMOGOROV_REYNOLDS_NUMBER,
        KOLMOGOROV_VISCOSITY,
        default_device,
    )


DEFAULT_DATA_PATH = Path(
    "/content/drive/MyDrive/Lab/CondDiff/NSE_Data_new.npy"
)


def _select_split(data: np.ndarray, split: str) -> np.ndarray:
    split_index = int(0.8 * len(data))
    if split == "train":
        return data[:split_index]
    if split == "test":
        return data[split_index:]
    if split == "all":
        return data
    raise ValueError(f"Unknown split: {split}")


def _select_centers(
    frame_count: int,
    max_windows: Optional[int],
    seed: int,
) -> np.ndarray:
    # PIDM's dataset admits centers 2, ..., frame_count - 3 because the public
    # dataset carries five frames, even though this residual uses only k-1:k+1.
    valid_centers = np.arange(2, frame_count - 2, dtype=np.int64)
    if valid_centers.size == 0:
        raise ValueError(
            "The selected split must contain at least five consecutive frames"
        )
    if max_windows is None or max_windows >= valid_centers.size:
        return valid_centers
    if max_windows <= 0:
        raise ValueError("max_windows must be positive")

    rng = np.random.default_rng(seed)
    selected = rng.choice(valid_centers, size=max_windows, replace=False)
    return np.sort(selected)


def evaluate_ground_truth_physics(
    data: np.ndarray,
    *,
    split: str = "train",
    batch_size: int = 64,
    max_windows: Optional[int] = None,
    seed: int = 0,
    dt: float = KOLMOGOROV_DT_PHYS,
    reynolds_number: float = KOLMOGOROV_REYNOLDS_NUMBER,
    viscosity: float = KOLMOGOROV_VISCOSITY,
    forcing_wavenumber: int = KOLMOGOROV_FORCING_WAVENUMBER,
    device: Optional[torch.device] = None,
) -> dict[str, float]:
    """Return global averages for the exact PIDM training residual."""
    if data.ndim != 3 or data.shape[1:] != (64, 64):
        raise ValueError(f"Expected data with shape (N,64,64), got {data.shape}")
    if np.iscomplexobj(data):
        raise ValueError("Expected real physical-space vorticity, got complex data")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if reynolds_number <= 0:
        raise ValueError("reynolds_number must be positive")
    if not math.isclose(
        viscosity,
        1.0 / reynolds_number,
        rel_tol=1e-7,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "viscosity must equal 1 / reynolds_number; "
            f"got viscosity={viscosity}, reynolds_number={reynolds_number}"
        )
    if forcing_wavenumber <= 0:
        raise ValueError("forcing_wavenumber must be positive")

    fields = _select_split(data, split)
    centers = _select_centers(len(fields), max_windows, seed)
    eval_device = device if device is not None else default_device()

    # The residual methods use only ``dt``, viscosity, and forcing wavenumber.
    # Constructing this minimal shell avoids allocating a U-Net or optimizer;
    # no learned parameters or checkpoint are involved in this calculation.
    residual_evaluator = DDPMTrainer.__new__(DDPMTrainer)
    residual_evaluator.dt = float(dt)
    residual_evaluator.cfg = SimpleNamespace(
        viscosity=float(viscosity),
        forcing_wavenumber=int(forcing_wavenumber),
    )

    temporal_sq_sum = 0.0
    rhs_sq_sum = 0.0
    residual_sq_sum = 0.0
    point_count = 0

    print("Ground-truth PIDM physics evaluation")
    print(f"  split:             {split}")
    print(f"  frames in split:   {len(fields)}")
    print(f"  evaluated windows: {len(centers)}")
    print(f"  device:            {eval_device}")
    print(f"  dt:                {dt}")
    print(f"  Reynolds number:   {reynolds_number}")
    print(f"  viscosity:         {viscosity}")
    print(f"  forcing wavenumber: {forcing_wavenumber}")

    with torch.no_grad():
        for start in range(0, len(centers), batch_size):
            center_batch = centers[start : start + batch_size]

            # Copy from the possibly read-only memory map before creating
            # tensors, then evaluate in float32 exactly as PIDM training does.
            omega_minus_1 = torch.from_numpy(
                np.array(fields[center_batch - 1], dtype=np.float32, copy=True)
            ).unsqueeze(1).to(eval_device)
            omega = torch.from_numpy(
                np.array(fields[center_batch], dtype=np.float32, copy=True)
            ).unsqueeze(1).to(eval_device)
            omega_plus_1 = torch.from_numpy(
                np.array(fields[center_batch + 1], dtype=np.float32, copy=True)
            ).unsqueeze(1).to(eval_device)

            temporal_derivative = (
                omega_plus_1 - omega_minus_1
            ) / (2.0 * dt)
            residual = residual_evaluator.pde_residual(
                omega_minus_1,
                omega,
                omega_plus_1,
            )
            # pde_residual defines residual = temporal_derivative - RHS.
            rhs = temporal_derivative - residual

            temporal_sq_sum += temporal_derivative.double().square().sum().item()
            rhs_sq_sum += rhs.double().square().sum().item()
            residual_sq_sum += residual.double().square().sum().item()
            point_count += residual.numel()

    temporal_mse = temporal_sq_sum / point_count
    rhs_mse = rhs_sq_sum / point_count
    residual_mse = residual_sq_sum / point_count
    metrics = {
        "gt_physics_loss_3frame": residual_mse,
        "temporal_derivative_mse": temporal_mse,
        "rhs_mse": rhs_mse,
        "residual_mse": residual_mse,
        "residual_rmse": math.sqrt(residual_mse),
        "evaluated_windows": float(len(centers)),
        "evaluated_points": float(point_count),
    }

    print("\nGlobal element-weighted averages")
    print(f"  gt_physics_loss_3frame:  {metrics['gt_physics_loss_3frame']:.9e}")
    print(f"  temporal_derivative_mse: {metrics['temporal_derivative_mse']:.9e}")
    print(f"  rhs_mse:                 {metrics['rhs_mse']:.9e}")
    print(f"  residual_mse:            {metrics['residual_mse']:.9e}")
    print(f"  residual_rmse:           {metrics['residual_rmse']:.9e}")
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PIDM's exact three-frame residual on ground-truth "
            "physical-space vorticity; no checkpoint is required."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--split",
        choices=("train", "test", "all"),
        default="train",
        help="Use the same first-80% training split by default.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optionally evaluate a deterministic random subset.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dt", type=float, default=KOLMOGOROV_DT_PHYS)
    parser.add_argument(
        "--reynolds-number",
        type=float,
        default=KOLMOGOROV_REYNOLDS_NUMBER,
    )
    parser.add_argument(
        "--viscosity",
        type=float,
        default=KOLMOGOROV_VISCOSITY,
    )
    parser.add_argument(
        "--forcing-wavenumber",
        type=int,
        default=KOLMOGOROV_FORCING_WAVENUMBER,
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
        help="Default: CUDA when available, otherwise CPU.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"Data file not found: {args.data}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")

    data = np.load(args.data, mmap_mode="r")
    evaluate_ground_truth_physics(
        data,
        split=args.split,
        batch_size=args.batch_size,
        max_windows=args.max_windows,
        seed=args.seed,
        dt=args.dt,
        reynolds_number=args.reynolds_number,
        viscosity=args.viscosity,
        forcing_wavenumber=args.forcing_wavenumber,
        device=torch.device(args.device) if args.device is not None else None,
    )


if __name__ == "__main__":
    main()
