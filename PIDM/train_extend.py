"""Continue a cfgConditional checkpoint with Kolmogorov physics loss.

Edit the values in the USER SETTINGS block, then run:

    python PIDM/train_extend.py

The source checkpoint is never overwritten because it is loaded from the
``pure_diffusion`` subdirectory. New ``conditional.pt`` and ``best.pt`` files
are written to CHECKPOINT_OUTPUT_DIR, while MLflow data is written to
MLFLOW_DIR.
"""

from pathlib import Path

import numpy as np

try:
    from .model import (
        KOLMOGOROV_DT_PHYS,
        KOLMOGOROV_FORCING_WAVENUMBER,
        KOLMOGOROV_REYNOLDS_NUMBER,
        KOLMOGOROV_VISCOSITY,
        run_training_resume,
    )
except ImportError:  # Direct execution: python PIDM/train_extend.py
    from model import (
        KOLMOGOROV_DT_PHYS,
        KOLMOGOROV_FORCING_WAVENUMBER,
        KOLMOGOROV_REYNOLDS_NUMBER,
        KOLMOGOROV_VISCOSITY,
        run_training_resume,
    )


# =============================================================================
# USER SETTINGS
# =============================================================================

# Final-epoch checkpoint produced by ddpm_sparse_cfg/cfgConditional.py. Do not
# use best.pt here: it can represent an earlier epoch and break the epoch-30
# data-loss handoff comparison.
CHECKPOINT_PATH = Path(
    "/content/drive/MyDrive/Lab/CondDiff/pure_diffusion/conditional.pt"
)

# Use the exact same ordered array that was passed to cfgConditional.py. The
# epoch-0 handoff check will stop before training if this data does not reproduce
# the data_loss stored in the source checkpoint.
DATA_PATH = Path(
    "/content/drive/MyDrive/Lab/CondDiff/NSE_Data_new.npy"
)

# Continued-training checkpoints and MLflow data are stored separately.
CHECKPOINT_OUTPUT_DIR = Path("/content/drive/MyDrive/Lab/CondDiff")
MLFLOW_DIR = Path("/content/drive/MyDrive/Lab/CondDiff/mlruns")

# Number of new physics-informed epochs to run.
ADDITIONAL_EPOCHS = 10

SEED = 0
RESTORE_OPTIMIZER = True

# Starts a new MLflow run by default. Set this to an existing run ID only when
# that run lives under MLFLOW_DIR and you intentionally want to append.
MLFLOW_RUN_ID = None
EPOCH_OFFSET = None

# Physics values from kolmogorov_demo.ipynb. These are passed explicitly so
# legacy values stored by cfgConditional.py cannot override them.
DT_PHYS = KOLMOGOROV_DT_PHYS
REYNOLDS_NUMBER = KOLMOGOROV_REYNOLDS_NUMBER
VISCOSITY = KOLMOGOROV_VISCOSITY
FORCING_WAVENUMBER = KOLMOGOROV_FORCING_WAVENUMBER
PHYSICS_T_MAX = 300  # Example value, adjust as needed

# Physics is enabled through adaptive scaling in model.py.
AUTO_SCALE_PHYSICS = True
PHYSICS_LOSS_TARGET_RATIO = 1.0


def main() -> None:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            "Set CHECKPOINT_PATH in train_extend.py to a cfgConditional.py "
            f"checkpoint. Current value: {CHECKPOINT_PATH}"
        )
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            "Set DATA_PATH in train_extend.py to the physical-space Kolmogorov "
            f"vorticity array. Current value: {DATA_PATH}"
        )
    if ADDITIONAL_EPOCHS <= 0:
        raise ValueError("ADDITIONAL_EPOCHS must be positive")

    # Copy-on-write mmap avoids loading the full trajectory into RAM while also
    # giving torch a writable NumPy view without modifying the source file.
    data = np.load(DATA_PATH, mmap_mode="c")
    if data.ndim != 3 or data.shape[1:] != (64, 64):
        raise ValueError(f"Expected data shape (N,64,64), got {data.shape}")
    if np.iscomplexobj(data):
        raise ValueError("Expected real physical-space vorticity, got complex data")

    print(f"Checkpoint:        {CHECKPOINT_PATH.resolve()}")
    print(f"Data:              {DATA_PATH.resolve()} {data.shape} {data.dtype}")
    print(f"Checkpoint output: {CHECKPOINT_OUTPUT_DIR.resolve()}")
    print(f"MLflow directory:  {MLFLOW_DIR.resolve()}")
    print(f"Additional epochs: {ADDITIONAL_EPOCHS}")
    print(
        "Physics:           "
        f"dt={DT_PHYS}, Re={REYNOLDS_NUMBER}, nu={VISCOSITY}, "
        f"forcing k={FORCING_WAVENUMBER}"
    )

    best_path, stats, cfg = run_training_resume(
        data=data,
        ckpt_path=str(CHECKPOINT_PATH),
        out_dir=str(CHECKPOINT_OUTPUT_DIR),
        additional_epochs=ADDITIONAL_EPOCHS,
        mlflow_dir=str(MLFLOW_DIR),
        mlflow_run_id=MLFLOW_RUN_ID,
        physics_t_max=PHYSICS_T_MAX,
        seed=SEED,
        epoch_offset=EPOCH_OFFSET,
        restore_optimizer=RESTORE_OPTIMIZER,
        dt_phys=DT_PHYS,
        reynolds_number=REYNOLDS_NUMBER,
        viscosity=VISCOSITY,
        forcing_wavenumber=FORCING_WAVENUMBER,
        auto_scale_physics=AUTO_SCALE_PHYSICS,
        physics_loss_target_ratio=PHYSICS_LOSS_TARGET_RATIO,
    )

    print(f"Finished. Best extended checkpoint: {best_path}")
    print(f"Normalization mean/std: {stats}")
    print(f"Continuation config: {cfg}")


if __name__ == "__main__":
    main()
