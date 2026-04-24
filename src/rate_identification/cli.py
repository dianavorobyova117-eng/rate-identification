"""CLI entry point for rate identification."""

import sys
from pathlib import Path

import click
import numpy as np
import pyulog
import yaml

from .data import (
    extract_segment_data,
    find_offboard_segment,
    get_acc_rates_data,
    get_angular_velocity_data,
)
from .identification import identify_axis
from .plots import (
    plot_data_filtering,
    plot_fit,
    plot_frequency_analysis,
)


@click.command()
@click.argument("ulg_file", type=click.Path(exists=True))
@click.option(
    "--duration",
    type=float,
    default=2.5,
    help="Identification duration in seconds (default: 2.5)",
)
@click.option(
    "--axes",
    type=click.Choice(["roll", "pitch", "both"]),
    default="both",
    help="Axes to identify (default: both)",
)
@click.option(
    "--max-delay",
    type=int,
    default=20,
    help="Maximum delay to search (default: 20)",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="output",
    help="Output directory (default: output)",
)
@click.option(
    "--bandwidth-hz",
    type=float,
    default=25.0,
    help="Analysis bandwidth in Hz (default: 25.0)",
)
def main(ulg_file: str, duration: float, axes: str, max_delay: int, output_dir: str, bandwidth_hz: float):
    """辨识PX4 ULog文件中的rate dynamics.

    自动检测offboard模式时长，并使用acc_rates_setpoint作为输入辨识roll/pitch动力学。
    """
    ulg_path = Path(ulg_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing: {ulg_path.name}")
    print(f"{'='*60}\n")

    # 加载ULog
    ulog = pyulog.ULog(str(ulg_path))

    # 检测offboard时段
    offboard = find_offboard_segment(ulog)
    if offboard is None:
        print("Error: No offboard mode detected in this log")
        sys.exit(1)

    offboard_start, offboard_end = offboard
    offboard_duration = offboard_end - offboard_start

    print(f"Offboard detected:")
    print(f"  Start: {offboard_start:.3f} s")
    print(f"  End: {offboard_end:.3f} s")
    print(f"  Duration: {offboard_duration:.3f} s")

    # 使用offboard起始时间，但使用指定的duration
    ident_duration = min(duration, offboard_duration)
    print(f"  Identification duration: {ident_duration:.3f} s\n")

    # 获取数据
    acc_rates = get_acc_rates_data(ulog)
    ang_vel = get_angular_velocity_data(ulog)

    if acc_rates is None or ang_vel is None:
        print("Error: Required topics not found")
        sys.exit(1)

    ts_acc, roll_sp, pitch_sp = acc_rates
    ts_ang, roll_meas, pitch_meas = ang_vel

    # 确定要处理的轴
    axes_to_process = []
    if axes in ["roll", "both"]:
        axes_to_process.append("roll")
    if axes in ["pitch", "both"]:
        axes_to_process.append("pitch")

    # 存储绘图数据
    plot_data = {
        "ts": {},
        "u": {},
        "y": {},
    }
    results = {}

    # 处理每个轴
    for axis in axes_to_process:
        print(f"[{axis.upper()}]")

        # 选择数据
        if axis == "roll":
            u_raw = roll_sp
            y_raw = roll_meas
        else:
            u_raw = pitch_sp
            y_raw = pitch_meas

        # 提取offboard后的数据
        ts_u, u = extract_segment_data(ts_acc, u_raw, offboard_start, ident_duration)
        ts_y, y = extract_segment_data(ts_ang, y_raw, offboard_start, ident_duration)

        # 去均值
        u = u - np.mean(u)
        y = y - np.mean(y)

        print(f"  Data points: {len(u)}")
        print(f"  Input std: {np.std(u):.4f} rad/s")
        print(f"  Output std: {np.std(y):.4f} rad/s")

        # 辨识多个阶数
        axis_fits = {}
        best_result = None
        best_fit = -1.0

        for order in [1, 2]:
            result = identify_axis(u, y, max_delay=max_delay, orders=[order])
            if result is not None:
                axis_fits[order] = {
                    "fit_pct": result.fit_pct,
                    "y_hat": result.y_hat,
                    "transfer_function": result.transfer_function,
                    "order": result.order,
                    "delay_samples": result.delay_samples,
                }

                if result.fit_pct > best_fit:
                    best_fit = result.fit_pct
                    best_result = result

        if best_result is None:
            print(f"  Error: Identification failed")
            continue

        print(f"  Best fit: {best_result.fit_pct:.1f}% (order={best_result.order}, delay={best_result.delay_samples})")
        print(f"  Stable: {best_result.stable}")
        print(f"  Transfer function: {best_result.transfer_function}\n")

        # 存储数据
        plot_data["ts"][axis] = ts_u
        plot_data["u"][axis] = u
        plot_data["y"][axis] = y

        results[axis] = {
            "u": u,
            "ts": ts_u,
            "fits": axis_fits,
            "best_order": best_result.order,
        }

    # 生成图片
    if results:
        stem = ulg_path.stem

        # 1. 数据过滤图
        plot_data_filtering(stem, plot_data["ts"], plot_data["u"], plot_data["y"],
                            output_path / f"{stem}_data_filtering.png")
        print(f"  Data filtering plot saved")

        # 2. 拟合图
        plot_fit(stem, results, output_path / f"{stem}_fit.png")
        print(f"  Fit plot saved")

        # 3. 频率分析图
        plot_frequency_analysis(stem, results, output_path / f"{stem}_bode.png", bandwidth_hz)
        print(f"  Frequency analysis plot saved\n")

        # 保存参数
        params = {
            "ulg_file": str(ulg_path),
            "offboard_start": float(offboard_start),
            "offboard_duration": float(offboard_duration),
            "identification_duration": float(ident_duration),
            "axes": {},
        }

        for axis, data in results.items():
            best_order = data["best_order"]
            best_fit_data = data["fits"][best_order]

            params["axes"][axis] = {
                "best_order": best_order,
                "transfer_function": best_fit_data["transfer_function"],
                "fit_percent": float(best_fit_data["fit_pct"]),
                "delay_samples": best_fit_data["delay_samples"],
            }

        params_file = output_path / f"{stem}_params.yaml"
        with open(params_file, "w") as f:
            yaml.dump(params, f, default_flow_style=False, sort_keys=False)

        print(f"Parameters saved: {params_file}")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
