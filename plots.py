"""Plotting helpers for training and experiment visualizations."""

from __future__ import annotations

import os
from typing import Iterable, List

import matplotlib.pyplot as plt

from train import moving_average


def plot_comparison_metric(
    fixed_values: List[float],
    decaying_values: List[float],
    ylabel: str,
    title: str,
    save_path: str,
    smoothing_window: int = 20,
) -> None:
    """Plot raw and smoothed curves for both experiment settings."""
    episodes_fixed = range(1, len(fixed_values) + 1)
    episodes_decay = range(1, len(decaying_values) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes_fixed, fixed_values, color="tab:blue", alpha=0.25, label="Fixed epsilon (raw)")
    plt.plot(episodes_decay, decaying_values, color="tab:orange", alpha=0.25, label="Decaying epsilon (raw)")

    fixed_smoothed = moving_average(fixed_values, window=smoothing_window)
    decay_smoothed = moving_average(decaying_values, window=smoothing_window)

    fixed_start = smoothing_window if len(fixed_values) >= smoothing_window else 1
    decay_start = smoothing_window if len(decaying_values) >= smoothing_window else 1

    plt.plot(
        range(fixed_start, fixed_start + len(fixed_smoothed)),
        fixed_smoothed,
        color="tab:blue",
        linewidth=2.2,
        label=f"Fixed epsilon ({smoothing_window}-episode average)",
    )
    plt.plot(
        range(decay_start, decay_start + len(decay_smoothed)),
        decay_smoothed,
        color="tab:orange",
        linewidth=2.2,
        label=f"Decaying epsilon ({smoothing_window}-episode average)",
    )

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_epsilon_decay(epsilon_values: Iterable[float], save_path: str) -> None:
    """Plot epsilon values recorded across episodes."""
    epsilon_values = list(epsilon_values)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(epsilon_values) + 1), epsilon_values, color="tab:green", linewidth=2)
    plt.title("Decaying Epsilon Across Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def output_path(project_dir: str, filename: str) -> str:
    """Create a clean save path inside the project directory."""
    return os.path.join(project_dir, filename)
