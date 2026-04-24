"""
Resume PIDM conditional DDPM training from a checkpoint.

Loads a previous checkpoint, restores model/optimizer/scaler/epoch, and continues
training via `run_training_resume` from PIDM `cfgConditional.py`.

MLflow continuation:
  - If --mlflow-run-id is provided, logging continues on that run.
  - Otherwise, if checkpoint contains mlflow_run_id, it is reused automatically.
  - Else, a new MLflow run is created.
"""
import argparse
import os
import sys
from pathlib import Path

def _find_project_root(start: Path) -> Path:
    """
    Find repo root containing PIDM/src by walking up from script location.
    Works for both:
      - <repo>/PIDM/resumeTraining.py
      - <repo>/PIDM/src/resumeTraining.py
    """
    cur = start.resolve()
    for candidate in [cur] + list(cur.parents):
        if (candidate / "PIDM" / "src").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not locate project root from {start}")


# Project root: .../2dNS_Conditional_Diffusion
_PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
_PIDM_SRC = _PROJECT_ROOT / "PIDM" / "src"
sys.path.insert(0, str(_PIDM_SRC))

from cfgConditional import run_training_resume  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Resume training from checkpoint (PIDM).")
    p.add_argument(
        "--data",
        type=str,
        default=str(_PROJECT_ROOT / "NSE_Data(Noisy).npy"),
        help="Path to (N, 64, 64) numpy data (same as original training).",
    )
    p.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Checkpoint .pt (best.pt or conditional.pt) from a previous PIDM training run.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output dir for checkpoints and mlruns (default: parent directory of --ckpt).",
    )
    p.add_argument(
        "--additional-epochs",
        type=int,
        default=10,
        help="Number of epochs to train after loading the checkpoint.",
    )
    p.add_argument(
        "--mlflow-run-id",
        type=str,
        default=None,
        help="Existing MLflow run ID to continue logging. If omitted, uses checkpoint run_id when available.",
    )
    p.add_argument(
        "--epoch-offset",
        type=int,
        default=None,
        help="Override last logged step: next train_loss step will be epoch_offset + 1. "
        "If omitted with --mlflow-run-id, max step is read from MLflow train_loss history.",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ckpt_path = os.path.abspath(args.ckpt)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = str(Path(ckpt_path).parent)
    out_dir = os.path.abspath(out_dir)

    import numpy as np

    data = np.load(args.data).astype(np.float32)
    print(f"Loaded data: {data.shape} from {args.data}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Output dir (checkpoints + mlruns): {out_dir}")

    final_path, stats, cfg = run_training_resume(
        data=data,
        ckpt_path=ckpt_path,
        out_dir=out_dir,
        additional_epochs=args.additional_epochs,
        mlflow_run_id=args.mlflow_run_id,
        seed=args.seed,
        epoch_offset=args.epoch_offset,
    )
    print("Done. Best checkpoint:", final_path, "stats:", stats)


if __name__ == "__main__":
    main()
