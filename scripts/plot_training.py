"""Plot Method 1 training curves from logs/method1/training_log.csv.

Usage:
    python scripts/plot_training.py
    python scripts/plot_training.py --log logs/method1/training_log.csv --out logs/method1/curves.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def smooth(series: pd.Series, window: int) -> pd.Series:
    if window <= 1 or len(series) < window:
        return series
    return series.rolling(window=window, min_periods=1, center=False).mean()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="logs/method1/training_log.csv")
    p.add_argument("--out", default=None,
                   help="Output PNG path (default: <log_dir>/curves.png)")
    p.add_argument("--smooth-window", type=int, default=10,
                   help="Rolling-mean window for the curves (rows, not steps).")
    args = p.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(f"Training log not found: {log_path}")
    df = pd.read_csv(log_path)
    if df.empty:
        print(f"Log file is empty: {log_path}")
        return 1

    print(f"Loaded {len(df)} rows from {log_path}")
    print(f"  step range: {df['step'].min():,} → {df['step'].max():,}")
    print(f"  episodes:   {int(df['episodes'].max()):,}")
    print(f"  final success_rate: {df['success_rate'].iloc[-1]:.2%}")
    print(f"  final mean_reward:  {df['mean_reward'].iloc[-1]:.3f}")

    out = Path(args.out) if args.out else log_path.parent / "curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    x = df["step"]
    w = args.smooth_window

    # Row 1: behavior
    axes[0, 0].plot(x, df["mean_reward"], alpha=0.25, color="C0")
    axes[0, 0].plot(x, smooth(df["mean_reward"], w), color="C0", linewidth=1.8)
    axes[0, 0].set_title("Mean episode return (last 100 eps)")
    axes[0, 0].set_ylabel("mean reward")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(x, df["success_rate"], alpha=0.25, color="C2")
    axes[0, 1].plot(x, smooth(df["success_rate"], w), color="C2", linewidth=1.8)
    axes[0, 1].set_title("Success rate (last 100 eps)")
    axes[0, 1].set_ylabel("success rate")
    axes[0, 1].set_ylim(-0.02, 1.02)
    axes[0, 1].grid(alpha=0.3)

    axes[0, 2].plot(x, df["episode_len"], alpha=0.25, color="C5")
    axes[0, 2].plot(x, smooth(df["episode_len"], w), color="C5", linewidth=1.8)
    axes[0, 2].set_title("Episode length (last 100 eps)")
    axes[0, 2].set_ylabel("steps")
    axes[0, 2].grid(alpha=0.3)

    # Row 2: optimization
    axes[1, 0].plot(x, df["policy_loss"], color="C3", linewidth=1.2)
    axes[1, 0].set_title("Policy loss")
    axes[1, 0].set_ylabel("loss")
    axes[1, 0].set_xlabel("env steps")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(x, df["value_loss"], color="C1", linewidth=1.2)
    axes[1, 1].set_title("Value loss")
    axes[1, 1].set_xlabel("env steps")
    axes[1, 1].grid(alpha=0.3)

    axes[1, 2].plot(x, df["entropy"], color="C4", linewidth=1.2)
    axes[1, 2].set_title("Policy entropy")
    axes[1, 2].set_xlabel("env steps")
    axes[1, 2].grid(alpha=0.3)

    fig.suptitle(f"Method 1 training — {log_path.relative_to(log_path.parent.parent)}",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
