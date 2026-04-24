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
    plot_final_result,
    plot_fit,
    plot_frequency_analysis,
)


def find_dominant_frequencies(
    u: np.ndarray, y: np.ndarray, dt: float, n_peaks: int = 3
) -> dict:
    """Find dominant frequencies using FFT.

    Args:
        u: Input signal
        y: Output signal
        dt: Sample time [s]
        n_peaks: Number of dominant peaks to return

    Returns:
        Dict with dominant frequencies for input and output
    """
    n = len(u)
    fs = 1.0 / dt

    # FFT
    fft_u = np.fft.rfft(u)
    fft_y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, d=dt)

    # Magnitude spectrum
    mag_u = np.abs(fft_u) / n
    mag_y = np.abs(fft_y) / n

    # Find peaks (skip DC component)
    def top_peaks(mag, freqs, n_peaks):
        # Skip DC (index 0)
        idx = np.argsort(mag[1:])[::-1][:n_peaks] + 1
        return [(float(freqs[i]), float(mag[i])) for i in idx]

    return {
        "input": top_peaks(mag_u, freqs, n_peaks),
        "output": top_peaks(mag_y, freqs, n_peaks),
        "fs": fs,
    }


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
def main(
    ulg_file: str,
    duration: float,
    axes: str,
    output_dir: str,
    bandwidth_hz: float,
):
    """Identify PX4 rate dynamics from ULog file.

    Uses continuous second-order model: H(s) = (b0 + b1*s) / (s^2 + a1*s + a2)
    """
    ulg_path = Path(ulg_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Processing: {ulg_path.name}")
    print(f"{'=' * 60}\n")

    # Load ULog
    ulog = pyulog.ULog(str(ulg_path))

    # Detect offboard segment
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

    # Use offboard start time with specified duration
    ident_duration = min(duration, offboard_duration)
    print(f"  Identification duration: {ident_duration:.3f} s\n")

    # Get data
    acc_rates = get_acc_rates_data(ulog)
    ang_vel = get_angular_velocity_data(ulog)

    if acc_rates is None or ang_vel is None:
        print("Error: Required topics not found")
        sys.exit(1)

    ts_acc, roll_sp, pitch_sp = acc_rates
    ts_ang, roll_meas, pitch_meas = ang_vel

    # Determine axes to process
    axes_to_process = []
    if axes in ["roll", "both"]:
        axes_to_process.append("roll")
    if axes in ["pitch", "both"]:
        axes_to_process.append("pitch")

    # Store plot data
    plot_data = {
        "ts": {},
        "u": {},
        "y": {},
    }
    results = {}

    # Process each axis
    for axis in axes_to_process:
        print(f"[{axis.upper()}]")

        # Select data
        if axis == "roll":
            u_raw = roll_sp
            y_raw = roll_meas
        else:
            u_raw = pitch_sp
            y_raw = pitch_meas

        # Extract offboard segment
        ts_u, u = extract_segment_data(ts_acc, u_raw, offboard_start, ident_duration)
        ts_y, y = extract_segment_data(ts_ang, y_raw, offboard_start, ident_duration)

        # Remove mean
        u = u - np.mean(u)
        y = y - np.mean(y)

        # Compute sample time from timestamps (in seconds)
        dt = float(np.median(np.diff(ts_u)))

        print(f"  Data points: {len(u)}")
        print(f"  Sample time: {dt:.4f} s ({1.0 / dt:.1f} Hz)")
        print(f"  Input std: {np.std(u):.4f} rad/s")
        print(f"  Output std: {np.std(y):.4f} rad/s")

        # Identify second-order model
        result = identify_axis(u, y, dt)

        if result is None:
            print(f"  Error: Identification failed")
            continue

        print(f"  Fit: {result.fit_pct:.1f}%")
        print(
            f"  omega_n: {result.omega_n:.2f} rad/s ({result.omega_n / (2 * np.pi):.2f} Hz)"
        )
        print(f"  zeta: {result.zeta:.4f}")
        print(f"  Stable: {result.stable}")
        print(f"  Transfer function: {result.transfer_function}")

        # FFT analysis
        fft_result = find_dominant_frequencies(u, y, dt)
        print(f"  Dominant frequencies (input):")
        for freq, mag in fft_result["input"]:
            print(f"    {freq:.2f} Hz (mag={mag:.4f})")
        print(f"  Dominant frequencies (output):")
        for freq, mag in fft_result["output"]:
            print(f"    {freq:.2f} Hz (mag={mag:.4f})")
        print()

        # Store data
        plot_data["ts"][axis] = ts_u
        plot_data["u"][axis] = u
        plot_data["y"][axis] = y

        results[axis] = {
            "u": u,
            "y": y,
            "ts": ts_u,
            "y_hat": result.y_hat,
            "fit_pct": result.fit_pct,
            "transfer_function": result.transfer_function,
            "omega_n": result.omega_n,
            "zeta": result.zeta,
            "poles": result.poles,
            "dominant_freqs": fft_result,
        }

    # Generate plots
    if results:
        stem = ulg_path.stem

        # 1. Data filtering plot
        plot_data_filtering(
            stem,
            plot_data["ts"],
            plot_data["u"],
            plot_data["y"],
            output_path / f"{stem}_data_filtering.png",
        )
        print(f"  Data filtering plot saved")

        # 2. Fit plot
        plot_fit(stem, results, output_path / f"{stem}_fit.png")
        print(f"  Fit plot saved")

        # 3. Frequency analysis plot
        plot_frequency_analysis(
            stem, results, output_path / f"{stem}_bode.png", bandwidth_hz
        )
        print(f"  Frequency analysis plot saved")

        # 4. Final result plot
        plot_final_result(stem, results, output_path / f"{stem}_final.png")
        print(f"  Final result plot saved\n")

        # Save parameters
        params = {
            "ulg_file": str(ulg_path),
            "offboard_start": float(offboard_start),
            "offboard_duration": float(offboard_duration),
            "identification_duration": float(ident_duration),
            "model": "H(s) = omega_n^2 / (s^2 + 2*zeta*omega_n*s + omega_n^2)",
            "axes": {},
        }

        for axis, data in results.items():
            freqs = data["dominant_freqs"]
            params["axes"][axis] = {
                "sample_rate_hz": float(freqs["fs"]),
                "transfer_function": data["transfer_function"],
                "fit_percent": float(data["fit_pct"]),
                "omega_n": float(data["omega_n"]),
                "omega_n_hz": float(data["omega_n"] / (2 * np.pi)),
                "zeta": float(data["zeta"]),
                "poles": [complex(p) for p in data["poles"]],
                "dominant_freqs_input": [
                    {"freq_hz": f, "magnitude": m} for f, m in freqs["input"]
                ],
                "dominant_freqs_output": [
                    {"freq_hz": f, "magnitude": m} for f, m in freqs["output"]
                ],
            }

        params_file = output_path / f"{stem}_params.yaml"
        with open(params_file, "w") as f:
            yaml.dump(params, f, default_flow_style=False, sort_keys=False)

        print(f"Parameters saved: {params_file}")

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
