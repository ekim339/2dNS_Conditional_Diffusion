"""Evaluate PIDM reconstructions and their Kolmogorov-flow residual.

For every selected center frame, the PIDM independently predicts the five
consecutive vorticity fields from k-2 through k+2 from their sparse observations.
The residual is computed only from those predictions. Ground-truth vorticity is
used only for reconstruction metrics.

The spatial right-hand side matches evaluate_continuous_2dns_kolmogorov.py and
PseudoSpectralNavierStokes2D. The temporal derivative uses the reference
evaluator's default five-point, fourth-order centered stencil.
"""

import argparse
import math
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from .model import (
        ConditionalDDPM,
        DDPMTrainer,
        DiffusionConfig,
        NavierStokesSparseDataset,
        default_device,
    )
except ImportError:  # Direct execution from PIDM/.
    from model import (
        ConditionalDDPM,
        DDPMTrainer,
        DiffusionConfig,
        NavierStokesSparseDataset,
        default_device,
    )


GRID_SIZE = (64, 64)
DT_SAVE = 0.001
REYNOLDS_NUMBER = 250.0
VISCOSITY = 0.004
FORCING_WAVENUMBER = 4


class KolmogorovResidual:
    """Repository-consistent residual for five predicted vorticity frames."""

    def __init__(
        self,
        dt: float = DT_SAVE,
        reynolds_number: float = REYNOLDS_NUMBER,
        viscosity: float = VISCOSITY,
        forcing_wavenumber: int = FORCING_WAVENUMBER,
    ):
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        if reynolds_number <= 0:
            raise ValueError(
                f"reynolds_number must be positive, got {reynolds_number}"
            )
        expected_viscosity = 1.0 / reynolds_number
        if not math.isclose(
            viscosity, expected_viscosity, rel_tol=1e-7, abs_tol=1e-12
        ):
            raise ValueError(
                "viscosity must equal 1 / reynolds_number for the referenced "
                f"solver; got viscosity={viscosity} and "
                f"reynolds_number={reynolds_number}"
            )

        self.dt = float(dt)
        self.reynolds_number = float(reynolds_number)
        self.viscosity = float(viscosity)
        self.forcing_wavenumber = int(forcing_wavenumber)
        self._operator_cache = {}

    def _operators(
        self,
        nx: int,
        ny: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the FlowConfig Fourier mesh, Laplacian, and forcing."""
        cache_key = (nx, ny, device.type, device.index, dtype)
        cached = self._operator_cache.get(cache_key)
        if cached is not None:
            return cached

        domain_length = 2.0 * math.pi
        kx_1d = torch.fft.fftfreq(nx, d=domain_length / nx, device=device)
        ky_1d = torch.fft.rfftfreq(ny, d=domain_length / ny, device=device)
        kx = kx_1d.to(dtype).reshape(1, nx, 1)
        ky = ky_1d.to(dtype).reshape(1, 1, ny // 2 + 1)
        double_derivative = -(2.0 * math.pi) ** 2 * (
            kx.square() + ky.square()
        )

        # FlowConfig.create_mesh() samples the forcing on endpoint-inclusive
        # linspace arrays, so reproduce that grid rather than arange(N)/N.
        x = torch.linspace(0.0, domain_length, nx, device=device, dtype=dtype)
        y = torch.linspace(0.0, domain_length, ny, device=device, dtype=dtype)
        _, y_grid = torch.meshgrid(x, y, indexing="ij")
        force_x = torch.sin(self.forcing_wavenumber * y_grid)
        force_y = torch.zeros_like(force_x)
        force_x_hat = torch.fft.rfft2(force_x)
        force_y_hat = torch.fft.rfft2(force_y)
        forcing_hat = (2j * math.pi) * (
            force_y_hat.unsqueeze(0) * kx - force_x_hat.unsqueeze(0) * ky
        )

        operators = (kx, ky, double_derivative, forcing_hat)
        self._operator_cache[cache_key] = operators
        return operators

    def repository_rhs(self, omega: torch.Tensor) -> torch.Tensor:
        """Return nonlinear_terms(omega_hat) + linear_terms(omega_hat)."""
        if omega.ndim != 3:
            raise ValueError(
                f"Expected omega with shape (B,H,W), got {omega.shape}"
            )

        batch_size, nx, ny = omega.shape
        if (nx, ny) != GRID_SIZE:
            raise ValueError(
                f"Expected vorticity fields with shape {GRID_SIZE}, got {(nx, ny)}"
            )

        kx, ky, double_derivative, forcing_hat = self._operators(
            nx, ny, omega.device, omega.dtype
        )
        omega_hat = torch.fft.rfft2(omega)

        safe_double_derivative = double_derivative.clone()
        safe_double_derivative[:, 0, 0] = 1.0
        psi_hat = -omega_hat / safe_double_derivative
        velocity_x_hat = (2j * math.pi) * ky * psi_hat
        velocity_y_hat = (-2j * math.pi) * kx * psi_hat
        velocity_x = torch.fft.irfft2(velocity_x_hat, s=(nx, ny))
        velocity_y = torch.fft.irfft2(velocity_y_hat, s=(nx, ny))

        grad_x = torch.fft.irfft2(
            (2j * math.pi) * kx * omega_hat, s=(nx, ny)
        )
        grad_y = torch.fft.irfft2(
            (2j * math.pi) * ky * omega_hat, s=(nx, ny)
        )
        advection_hat = torch.fft.rfft2(
            -(grad_x * velocity_x + grad_y * velocity_y)
        )

        # This intentionally has no dealiasing mask. The referenced repository's
        # utils.dealiasing() returns its original JAX array because its .at[].set
        # results are not rebound, which is the behavior used by the evaluator.
        diffusion_hat = self.viscosity * double_derivative * omega_hat
        rhs_hat = (
            advection_hat
            + forcing_hat.expand(batch_size, -1, -1)
            + diffusion_hat
        )
        return torch.fft.irfft2(rhs_hat, s=(nx, ny))

    def __call__(
        self,
        omega_k_minus_2: torch.Tensor,
        omega_k_minus_1: torch.Tensor,
        omega_k: torch.Tensor,
        omega_k_plus_1: torch.Tensor,
        omega_k_plus_2: torch.Tensor,
    ) -> torch.Tensor:
        """Return the five-point temporal residual centered at omega[k]."""
        fields = (
            omega_k_minus_2,
            omega_k_minus_1,
            omega_k,
            omega_k_plus_1,
            omega_k_plus_2,
        )
        if any(field.ndim != 4 or field.shape[1] != 1 for field in fields):
            shapes = [tuple(field.shape) for field in fields]
            raise ValueError(
                f"Expected five (B,1,H,W) vorticity tensors, got {shapes}"
            )
        if any(field.shape != omega_k.shape for field in fields):
            raise ValueError("All five predicted vorticity fields must match")

        w_minus_2, w_minus_1, w_cur, w_plus_1, w_plus_2 = (
            field.squeeze(1) for field in fields
        )
        d_omega_dt = (
            w_minus_2 - 8.0 * w_minus_1 + 8.0 * w_plus_1 - w_plus_2
        ) / (12.0 * self.dt)
        return d_omega_dt - self.repository_rhs(w_cur)


def _sample_subset(
    dataset: Dataset,
    count: int,
    rng: np.random.Generator,
) -> Tuple[Subset, np.ndarray]:
    """Sample center indices without breaking their consecutive frame windows."""
    if count <= 0:
        raise ValueError(f"sample count must be positive, got {count}")
    selected_count = min(count, len(dataset))
    indices = np.sort(rng.choice(len(dataset), selected_count, replace=False))
    return Subset(dataset, indices.tolist()), indices


@torch.no_grad()
def evaluate_on_test(
    trainer: DDPMTrainer,
    data_loader: DataLoader,
    mean: float,
    std: float,
    residual_evaluator: KolmogorovResidual,
    num_batches: Optional[int] = None,
    guidance_scale: float = 1.0,
) -> Dict[str, float]:
    """Generate predicted five-frame windows and report evaluation metrics."""
    device = trainer.device
    trainer.model.eval()

    total_samples = 0
    field_sq_sum = 0.0
    field_abs_sum = 0.0
    field_points = 0
    sensor_sq_sum = 0.0
    sensor_points = 0.0
    residual_sq_sum = 0.0
    residual_sum = 0.0
    residual_abs_sum = 0.0
    residual_absmax = 0.0
    residual_points = 0

    print(f"\n{'=' * 60}")
    print("Starting PIDM Evaluation")
    print(f"{'=' * 60}")
    print(
        "Batches: "
        + (
            str(num_batches)
            if num_batches is not None
            else f"all ({len(data_loader)})"
        )
    )
    print("Residual input: five PIDM-predicted consecutive frames")
    print("Time derivative: 5-point, 4th-order centered")
    print(
        f"dt={residual_evaluator.dt}, "
        f"Re={residual_evaluator.reynolds_number:g}, "
        f"nu={residual_evaluator.viscosity:g}, "
        f"forcing k={residual_evaluator.forcing_wavenumber}"
    )
    print(f"Guidance scale: {guidance_scale}")
    print(f"Device: {device}")
    print(f"{'=' * 60}\n")

    start_time = time.time()
    for batch_index, batch in enumerate(data_loader):
        if num_batches is not None and batch_index >= num_batches:
            break
        if len(batch) != 6:
            raise ValueError(
                f"Expected six tensors from the PIDM dataset, got {len(batch)}"
            )

        (
            _omega_before_norm,
            omega_norm,
            _omega_after_norm,
            cond_before,
            cond,
            cond_after,
        ) = batch

        # model.py packs all four neighboring conditions as [k-2,k-1] and
        # [k+1,k+2], allowing the evaluator to generate the full stencil.
        if cond_before.ndim != 5 or cond_before.shape[1:3] != (2, 2):
            raise ValueError(
                "Expected cond_before with shape (B,2,2,64,64), got "
                f"{tuple(cond_before.shape)}"
            )
        if cond_after.ndim != 5 or cond_after.shape[1:3] != (2, 2):
            raise ValueError(
                "Expected cond_after with shape (B,2,2,64,64), got "
                f"{tuple(cond_after.shape)}"
            )
        cond_minus_2 = cond_before[:, 0].to(device)
        cond_minus_1 = cond_before[:, 1].to(device)
        cond = cond.to(device)
        cond_plus_1 = cond_after[:, 0].to(device)
        cond_plus_2 = cond_after[:, 1].to(device)
        omega_norm = omega_norm.to(device)

        batch_size = omega_norm.shape[0]
        sample_start = time.time()
        omega_minus_2_pred_norm = trainer.sample_cfg(
            cond=cond_minus_2,
            guidance_scale=guidance_scale,
            shape=tuple(omega_norm.shape),
        )
        omega_minus_1_pred_norm = trainer.sample_cfg(
            cond=cond_minus_1,
            guidance_scale=guidance_scale,
            shape=tuple(omega_norm.shape),
        )
        omega_pred_norm = trainer.sample_cfg(
            cond=cond,
            guidance_scale=guidance_scale,
            shape=tuple(omega_norm.shape),
        )
        omega_plus_1_pred_norm = trainer.sample_cfg(
            cond=cond_plus_1,
            guidance_scale=guidance_scale,
            shape=tuple(omega_norm.shape),
        )
        omega_plus_2_pred_norm = trainer.sample_cfg(
            cond=cond_plus_2,
            guidance_scale=guidance_scale,
            shape=tuple(omega_norm.shape),
        )
        sample_time = time.time() - sample_start

        scale = std + 1e-8
        omega_true = omega_norm * scale + mean
        omega_minus_2_pred = omega_minus_2_pred_norm * scale + mean
        omega_minus_1_pred = omega_minus_1_pred_norm * scale + mean
        omega_pred = omega_pred_norm * scale + mean
        omega_plus_1_pred = omega_plus_1_pred_norm * scale + mean
        omega_plus_2_pred = omega_plus_2_pred_norm * scale + mean

        # The residual depends only on generated fields, never ground truth.
        residual = residual_evaluator(
            omega_minus_2_pred.float(),
            omega_minus_1_pred.float(),
            omega_pred.float(),
            omega_plus_1_pred.float(),
            omega_plus_2_pred.float(),
        )

        field_error = omega_pred - omega_true
        mask = cond[:, 1:2]
        field_sq_sum += field_error.double().square().sum().item()
        field_abs_sum += field_error.double().abs().sum().item()
        field_points += field_error.numel()
        sensor_sq_sum += (
            field_error.double().square() * mask.double()
        ).sum().item()
        sensor_points += mask.double().sum().item()

        residual_double = residual.double()
        residual_sq_sum += residual_double.square().sum().item()
        residual_sum += residual_double.sum().item()
        residual_abs_sum += residual_double.abs().sum().item()
        residual_absmax = max(
            residual_absmax, residual_double.abs().max().item()
        )
        residual_points += residual.numel()
        total_samples += batch_size

        batch_mse = field_error.square().mean().item()
        batch_residual_rmse = residual.square().mean().sqrt().item()
        print(
            f"  Batch {batch_index + 1}/{len(data_loader)} | "
            f"samples={batch_size} | center MSE={batch_mse:.6e} | "
            f"residual RMSE={batch_residual_rmse:.6e} | "
            f"sampling={sample_time:.1f}s"
        )

    if total_samples == 0:
        raise ValueError("No evaluation batches were processed")

    total_time = time.time() - start_time
    physics_mse = residual_sq_sum / residual_points
    metrics = {
        "mse": field_sq_sum / field_points,
        "mae": field_abs_sum / field_points,
        "sensor_mse": sensor_sq_sum / sensor_points,
        "physics_mse": physics_mse,
        "residual_rmse": math.sqrt(physics_mse),
        "residual_mean": residual_sum / residual_points,
        "residual_absmean": residual_abs_sum / residual_points,
        "residual_absmax": residual_absmax,
        "total_samples": total_samples,
        "total_time": total_time,
        "residual_points": residual_points,
        "_residual_sq_sum": residual_sq_sum,
        "_residual_sum": residual_sum,
        "_residual_abs_sum": residual_abs_sum,
    }

    print(f"\n{'=' * 60}")
    print("PIDM Evaluation Complete")
    print(f"{'=' * 60}")
    print(f"Samples evaluated:       {total_samples}")
    print(f"Full-field MSE:          {metrics['mse']:.9e}")
    print(f"Full-field MAE:          {metrics['mae']:.9e}")
    print(f"Sensor MSE:              {metrics['sensor_mse']:.9e}")
    print(f"Physics MSE:             {metrics['physics_mse']:.9e}")
    print(f"Residual RMSE:           {metrics['residual_rmse']:.9e}")
    print(f"Residual mean:           {metrics['residual_mean']:.9e}")
    print(f"Residual |mean|:         {metrics['residual_absmean']:.9e}")
    print(f"Residual |max|:          {metrics['residual_absmax']:.9e}")
    print(f"Total time:              {total_time:.1f}s")
    print(f"Time per five-frame window: {total_time / total_samples:.2f}s")
    print(f"{'=' * 60}\n")
    return metrics


def _config_from_checkpoint(cfg_dict: Dict) -> DiffusionConfig:
    fields = set(DiffusionConfig.__dataclass_fields__)
    return DiffusionConfig(
        **{key: value for key, value in cfg_dict.items() if key in fields}
    )


def run_eval(
    ckpt_path: str,
    data_path: str,
    batch_size: int = 16,
    num_batches: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    samples_per_split: int = 100,
    sensor_stride: int = 8,
    seed: int = 42,
    dt: float = DT_SAVE,
    reynolds_number: float = REYNOLDS_NUMBER,
    viscosity: float = VISCOSITY,
    forcing_wavenumber: int = FORCING_WAVENUMBER,
) -> Dict[str, Dict[str, float]]:
    """Evaluate randomly selected, temporally intact PIDM frame windows."""
    device = default_device()
    print("Device:", device)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    if not ckpt_path.endswith(".pt"):
        raise ValueError(f"Checkpoint must be a .pt file, got: {ckpt_path}")
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Copy-on-write mapping avoids eagerly copying the approximately 1.5 GiB
    # dataset while still exposing a writable NumPy buffer to Torch.
    data = np.load(data_path, mmap_mode="c")
    if data.ndim != 3 or data.shape[1:] != GRID_SIZE:
        raise ValueError(f"Expected data shape (N,64,64), got {data.shape}")
    if np.iscomplexobj(data):
        raise ValueError("Expected real physical-space vorticity, got complex data")
    data_t = torch.from_numpy(data)
    if not data_t.is_floating_point():
        data_t = data_t.float()

    total_frames = data_t.shape[0]
    train_frames = int(0.8 * total_frames)
    train_full = data_t[:train_frames]
    test_full = data_t[train_frames:]
    print(
        f"Loaded {data.shape} {data.dtype}: {train_frames} train frames, "
        f"{total_frames - train_frames} test frames"
    )

    try:
        checkpoint = torch.load(
            ckpt_path, map_location=device, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=device)

    mean = float(checkpoint["train_mean"])
    std = float(checkpoint["train_std"])
    cfg_dict = checkpoint["cfg"]
    if guidance_scale is None:
        guidance_scale = float(cfg_dict.get("guidance_scale", 1.0))

    model = ConditionalDDPM(
        T=cfg_dict["T"], emb_dim=256, base_ch=64
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    cfg = _config_from_checkpoint(cfg_dict)
    # Sampling does not use these PDE values, but current model.py validates
    # their consistency in DDPMTrainer. This also permits evaluation of older
    # PIDM checkpoints whose stored diagnostic viscosity preceded Re=250 data.
    cfg.dt_phys = dt
    cfg.reynolds_number = reynolds_number
    cfg.viscosity = viscosity
    cfg.forcing_wavenumber = forcing_wavenumber
    trainer = DDPMTrainer(
        model, cfg, device, data_mean=mean, data_std=std
    )

    train_windows = NavierStokesSparseDataset(
        train_full, mean=mean, std=std, sensor_stride=sensor_stride
    )
    test_windows = NavierStokesSparseDataset(
        test_full, mean=mean, std=std, sensor_stride=sensor_stride
    )
    rng = np.random.default_rng(seed)
    train_subset, train_indices = _sample_subset(
        train_windows, samples_per_split, rng
    )
    test_subset, test_indices = _sample_subset(
        test_windows, samples_per_split, rng
    )
    print(
        f"Selected {len(train_indices)} train and {len(test_indices)} test "
        "center frames without breaking temporal adjacency"
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    train_loader = DataLoader(train_subset, **loader_kwargs)
    test_loader = DataLoader(test_subset, **loader_kwargs)
    residual_evaluator = KolmogorovResidual(
        dt=dt,
        reynolds_number=reynolds_number,
        viscosity=viscosity,
        forcing_wavenumber=forcing_wavenumber,
    )

    print(f"\n{'=' * 60}\nEVALUATING TRAIN SAMPLES\n{'=' * 60}")
    train_metrics = evaluate_on_test(
        trainer,
        train_loader,
        mean,
        std,
        residual_evaluator,
        num_batches=num_batches,
        guidance_scale=guidance_scale,
    )
    print(f"\n{'=' * 60}\nEVALUATING TEST SAMPLES\n{'=' * 60}")
    test_metrics = evaluate_on_test(
        trainer,
        test_loader,
        mean,
        std,
        residual_evaluator,
        num_batches=num_batches,
        guidance_scale=guidance_scale,
    )

    sample_count = train_metrics["total_samples"] + test_metrics["total_samples"]
    residual_points = (
        train_metrics["residual_points"] + test_metrics["residual_points"]
    )
    physics_mse = (
        train_metrics["_residual_sq_sum"]
        + test_metrics["_residual_sq_sum"]
    ) / residual_points
    residual_mean = (
        train_metrics["_residual_sum"] + test_metrics["_residual_sum"]
    ) / residual_points
    residual_absmean = (
        train_metrics["_residual_abs_sum"]
        + test_metrics["_residual_abs_sum"]
    ) / residual_points

    def sample_weighted(metric_name: str) -> float:
        return (
            train_metrics[metric_name] * train_metrics["total_samples"]
            + test_metrics[metric_name] * test_metrics["total_samples"]
        ) / sample_count

    combined_metrics = {
        "mse": sample_weighted("mse"),
        "mae": sample_weighted("mae"),
        "sensor_mse": sample_weighted("sensor_mse"),
        "physics_mse": physics_mse,
        "residual_rmse": math.sqrt(physics_mse),
        "residual_mean": residual_mean,
        "residual_absmean": residual_absmean,
        "residual_absmax": max(
            train_metrics["residual_absmax"],
            test_metrics["residual_absmax"],
        ),
        "total_samples": sample_count,
        "total_time": train_metrics["total_time"] + test_metrics["total_time"],
    }

    print(f"\n{'=' * 60}")
    print(f"COMBINED PIDM RESULTS ({sample_count} predicted five-frame windows)")
    print(f"{'=' * 60}")
    print(f"Full-field MSE:  {combined_metrics['mse']:.9e}")
    print(f"Full-field MAE:  {combined_metrics['mae']:.9e}")
    print(f"Sensor MSE:      {combined_metrics['sensor_mse']:.9e}")
    print(f"Physics MSE:     {combined_metrics['physics_mse']:.9e}")
    print(f"Residual RMSE:   {combined_metrics['residual_rmse']:.9e}")
    print(f"Residual mean:   {combined_metrics['residual_mean']:.9e}")
    print(f"Residual |mean|: {combined_metrics['residual_absmean']:.9e}")
    print(f"Residual |max|:  {combined_metrics['residual_absmax']:.9e}")
    print(f"Guidance scale:  {guidance_scale}")
    print(f"{'=' * 60}\n")
    return {
        "train": train_metrics,
        "test": test_metrics,
        "combined": combined_metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PIDM reconstructions and the repository-consistent "
            "Kolmogorov residual of five predicted frames."
        )
    )
    parser.add_argument(
        "--checkpoint", required=True, help="PIDM checkpoint produced by model.py"
    )
    parser.add_argument(
        "--data",
        default=(
            "/Users/eugenekim/AIMS Lab/Controlling-Kolmogorov-Flow/"
            "kolmogorov_Re_250_vorticity.npy"
        ),
        help="Real physical-space vorticity array with shape (N,64,64)",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--num-batches",
        type=int,
        default=None,
        help="Batches per split; omit to evaluate every selected center",
    )
    parser.add_argument("--samples-per-split", type=int, default=100)
    parser.add_argument("--sensor-stride", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--dt", type=float, default=DT_SAVE)
    parser.add_argument("--reynolds-number", type=float, default=REYNOLDS_NUMBER)
    parser.add_argument("--viscosity", type=float, default=VISCOSITY)
    parser.add_argument(
        "--forcing-wavenumber", type=int, default=FORCING_WAVENUMBER
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(
        ckpt_path=args.checkpoint,
        data_path=args.data,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        guidance_scale=args.guidance_scale,
        samples_per_split=args.samples_per_split,
        sensor_stride=args.sensor_stride,
        seed=args.seed,
        dt=args.dt,
        reynolds_number=args.reynolds_number,
        viscosity=args.viscosity,
        forcing_wavenumber=args.forcing_wavenumber,
    )
