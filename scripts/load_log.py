#!/usr/bin/env python3
"""Load psi-inference_rtc binary logs and plot torso action channels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


LOG_SPECS = {
    "pd_targets": ("pd_targets.bin", 58),
    "raw_actions": ("raw_actions.bin", 36),
    "ik_extra_hist": ("ik_extra_hist.bin", 1043),
}

RAW_ACTION_SLICE = slice(28, 32)
RAW_ACTION_LABELS = ("torso_roll", "torso_pitch", "torso_yaw", "torso_height")


def log_dtype(dim: int) -> np.dtype:
    return np.dtype([
        ("t_ns", np.int64),
        ("x", np.float32, (dim,)),
    ])


def load_log(path: Path, dim: int) -> np.ndarray:
    dtype = log_dtype(dim)
    if not path.exists():
        raise FileNotFoundError(f"missing log file: {path}")

    size = path.stat().st_size
    if size % dtype.itemsize != 0:
        raise ValueError(
            f"{path} size {size} is not a multiple of row size {dtype.itemsize}; "
            "the log may be truncated"
        )

    return np.fromfile(path, dtype=dtype)


def load_all_logs(log_dir: Path) -> dict[str, np.ndarray]:
    logs = {}
    for name, (filename, dim) in LOG_SPECS.items():
        logs[name] = load_log(log_dir / filename, dim)
    return logs


def elapsed_seconds(rows: np.ndarray) -> np.ndarray:
    if len(rows) == 0:
        return np.array([], dtype=np.float64)
    return (rows["t_ns"].astype(np.float64) - float(rows["t_ns"][0])) * 1e-9


def plot_raw_action_torso(raw_actions: np.ndarray, output: Path | None, show: bool) -> None:
    import matplotlib.pyplot as plt

    if len(raw_actions) == 0:
        raise ValueError("raw_actions log is empty")

    t = elapsed_seconds(raw_actions)
    y = raw_actions["x"][:, RAW_ACTION_SLICE]

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(11, 8), constrained_layout=True)
    for i, (axis, label) in enumerate(zip(axes, RAW_ACTION_LABELS)):
        axis.plot(t, y[:, i], linewidth=1.3)
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)

    axes[-1].set_xlabel("time since first raw action (s)")
    fig.suptitle("raw_actions[28:32] over time")

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160)
        print(f"saved plot: {output}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load psi-inference_rtc .bin logs and plot raw_actions[28:32]."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory containing pd_targets.bin, raw_actions.bin, and ik_extra_hist.bin.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/raw_actions_28_32.png"),
        help="Path to save the plot. Use --output '' to disable saving.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window after loading the logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output if str(args.output) else None

    logs = load_all_logs(args.log_dir)
    for name, rows in logs.items():
        print(f"{name}: {len(rows)} rows")

    plot_raw_action_torso(logs["raw_actions"], output=output, show=args.show)


if __name__ == "__main__":
    main()
