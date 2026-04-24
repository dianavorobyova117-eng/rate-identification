"""Plotting functions for identification results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def plot_data_filtering(
    ulg_name: str,
    ts: dict[str, np.ndarray],
    u: dict[str, np.ndarray],
    y: dict[str, np.ndarray],
    output_path: Path,
):
    """绘制归一化输入输出数据."""
    n_axes = len(ts)
    if n_axes == 0:
        return

    fig, axes = plt.subplots(n_axes, 1, figsize=(12, 4.5 * n_axes), squeeze=False)

    for idx, (axis_name, axis_ts) in enumerate(ts.items()):
        ax = axes[idx, 0]

        u_norm = (u[axis_name] - np.mean(u[axis_name])) / (np.std(u[axis_name]) + 1e-12)
        y_norm = (y[axis_name] - np.mean(y[axis_name])) / (np.std(y[axis_name]) + 1e-12)

        ax.plot(axis_ts, u_norm, color="tab:blue", lw=1.2, label="input u (norm)")
        ax.plot(axis_ts, y_norm, color="tab:orange", lw=1.2, label="output y (norm)")

        ax.set_title(f"{ulg_name} - {axis_name.upper()} segment")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Normalized amplitude")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize="small")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_fit(
    ulg_name: str,
    results: dict[str, dict],
    output_path: Path,
):
    """绘制时域拟合对比图."""
    n_axes = len(results)
    if n_axes == 0:
        return

    fig, axes = plt.subplots(n_axes, 1, figsize=(12, 4.5 * n_axes), squeeze=False)
    colors = {1: "tab:blue", 2: "tab:orange"}

    for idx, (axis_name, data) in enumerate(results.items()):
        ax = axes[idx, 0]

        ts = data["ts"]
        u = data["u"]
        y_centered = u  # 已去均值的输出

        ax.plot(ts, y_centered, color="k", lw=1.2, label="measured y")

        for order, fit_data in sorted(data["fits"].items()):
            label = f"{order}-order (fit={fit_data['fit_pct']:.1f}%)"
            if order == data["best_order"]:
                label += " *"

            ax.plot(
                ts,
                fit_data["y_hat"],
                color=colors.get(order, "tab:green"),
                lw=1.4,
                ls="--",
                label=label,
            )

        ax.set_title(f"{ulg_name} - {axis_name.upper()} fit")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Rate (rad/s)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize="small")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_frequency_analysis(
    ulg_name: str,
    results: dict[str, dict],
    output_path: Path,
    bandwidth_hz: float = 25.0,
):
    """绘制频域分析图（波特图）."""
    n_axes = len(results)
    if n_axes == 0:
        return

    fig, axes = plt.subplots(n_axes, 3, figsize=(15, 4.2 * n_axes), squeeze=False)
    colors = {1: "tab:blue", 2: "tab:orange"}

    for row, (axis_name, data) in enumerate(results.items()):
        ax_err = axes[row, 0]
        ax_phase = axes[row, 1]
        ax_coh = axes[row, 2]

        # 计算频率响应
        u = data["u"]
        y = u  # 输出即输入（已去均值）

        for order, fit_data in sorted(data["fits"].items()):
            y_hat = fit_data["y_hat"]
            label = f"{order}-order"
            if order == data["best_order"]:
                label += " *"

            color = colors.get(order, "tab:green")

            # 计算频域指标
            u_c = u - np.mean(u)
            y_c = y - np.mean(y)
            y_hat_c = y_hat - np.mean(y_hat)

            dt = float(np.median(np.diff(data["ts"]))) if len(data["ts"]) > 1 else 1.0 / 50.0
            fs = 1.0 / dt
            nperseg = max(32, min(256, len(u_c) // 4))
            noverlap = nperseg // 2

            freq, coh = signal.coherence(u_c, y_c, fs=fs, nperseg=nperseg, noverlap=noverlap)
            _, pyy = signal.welch(y_c, fs=fs, nperseg=nperseg, noverlap=noverlap)
            _, s_yhat_y = signal.csd(y_hat_c, y_c, fs=fs, nperseg=nperseg, noverlap=noverlap)

            ratio = np.divide(s_yhat_y, pyy, out=np.zeros_like(s_yhat_y), where=np.abs(pyy) > 1e-15)
            rel_error_mag_db = 20.0 * np.log10(np.abs(ratio - 1.0) + 1e-15)
            phase_bias_deg = np.degrees(np.unwrap(np.angle(ratio)))

            mask = (freq >= 0.1) & (freq <= bandwidth_hz)

            ax_err.semilogx(freq[mask], rel_error_mag_db[mask], color=color, lw=1.4, label=label)
            ax_phase.semilogx(freq[mask], phase_bias_deg[mask], color=color, lw=1.4, label=label)
            ax_coh.semilogx(freq, coh, color=color, lw=1.2, label=label)

        ax_err.axhline(0.0, color="gray", ls="--", alpha=0.4)
        ax_phase.axhline(0.0, color="gray", ls="--", alpha=0.4)
        ax_coh.axhline(0.6, color="gray", ls="--", alpha=0.6, label="coh>0.6")

        ax_err.set_title(f"{axis_name.upper()} relative error")
        ax_phase.set_title(f"{axis_name.upper()} phase bias")
        ax_coh.set_title(f"{axis_name.upper()} coherence (u->y)")

        ax_err.set_ylabel("Error (dB)")
        ax_phase.set_ylabel("Phase bias (deg)")
        ax_coh.set_ylabel("Coherence")

        for ax in (ax_err, ax_phase, ax_coh):
            ax.set_xlabel("Frequency (Hz)")
            ax.set_xlim(0.1, bandwidth_hz * 1.1)
            ax.grid(True, alpha=0.3, which="both")
            ax.legend(loc="upper right", fontsize="small")

    fig.suptitle(f"Frequency Analysis: {ulg_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_transfer_function(
    result: dict,
    axis_name: str,
    output_path: Path,
):
    """绘制传递函数表达式."""
    fig, ax = plt.subplots(figsize=(10, 3))

    best_fit = result["fits"][result["best_order"]]
    text = f"{axis_name.upper()} Transfer Function:\n\n{best_fit['transfer_function']}"

    ax.text(0.5, 0.5, text, ha="center", va="center",
            fontsize=12, family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
