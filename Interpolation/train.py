import argparse
from pathlib import Path

from interpolation import run_interpolation


def parse_args():
    project_root = Path(__file__).resolve().parent.parent
    default_data = project_root / "/content/drive/MyDrive/Lab/CondDiff/NSE_Data(Noisy).npy"
    default_out = project_root / "/content/drive/MyDrive/Lab/CondDiff/"

    p = argparse.ArgumentParser(description="Run direct spatial interpolation baseline.")
    p.add_argument("--data", type=str, default=str(default_data), help="Path to data .npy (N,64,64)")
    p.add_argument("--out-dir", type=str, default=str(default_out), help="Output directory for metrics/results")
    p.add_argument(
        "--method",
        type=str,
        default="bilinear",
        choices=["nearest", "bilinear", "bicubic", "linear", "cubic"],
        help="Interpolation method",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=None, help="Optional debug limit per split")
    p.add_argument("--save-predictions", action="store_true", help="Save NPZ predictions/targets")
    p.add_argument("--epochs", type=int, default=30, help="Number of epochs to run")
    p.add_argument("--resume", action="store_true", help="Resume from state checkpoint")
    p.add_argument("--resume-state-path", type=str, default=None, help="Optional resume state JSON path")
    p.add_argument("--mlflow-run-id", type=str, default=None, help="Optional MLflow run id to continue")
    p.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run_interpolation(
        data_path=args.data,
        out_dir=args.out_dir,
        method=args.method,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
        save_predictions=args.save_predictions,
        use_mlflow=not args.no_mlflow,
        epochs=args.epochs,
        resume=args.resume,
        resume_state_path=args.resume_state_path,
        mlflow_run_id=args.mlflow_run_id,
    )
    combined = summary["splits"]["combined"]
    print(
        "\nDone | "
        f"MSE={combined['mse']:.6e} | "
        f"MAE={combined['mae']:.6e} | "
        f"Sensor MSE={combined['sensor_mse']:.6e}"
    )
