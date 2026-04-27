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
    get_thrust_acceleration_setpoint_data,
    get_accel_z_data,
)
from .identification import identify_axis, fit_magnitude_phase_percent
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
    help="Axes to identify (default: both, only for rate mode)",
)
@click.option(
    "--mode",
    type=click.Choice(["rate", "accel"]),
    default="rate",
    help="Identification mode: rate (angular velocity) or accel (thrust acceleration)",
)
@click.option(
    "--model-order",
    type=click.Choice(["1", "2"]),
    default="2",
    help="Model order: 1 (first order) or 2 (second order, default)",
)
@click.option(
    "--max-delay",
    type=int,
    default=30,
    help="Maximum delay to search in samples (default: 30, ~0.6s at 50Hz)",
)
@click.option(
    "--robust-loss",
    type=click.Choice(["linear", "huber", "cauchy", "soft_l1", "arctan"]),
    default="linear",
    help="Robust loss function for outlier rejection (default: linear)",
)
@click.option(
    "--f-scale",
    type=float,
    default=None,
    help="Soft margin for robust loss in rad/s (default: auto-detect)",
)
@click.option(
    "--phase-loss",
    is_flag=True,
    default=False,
    help="Use phase-based loss function (optimizes phase matching instead of magnitude)",
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
    mode: str,
    model_order: str,
    max_delay: int,
    robust_loss: str,
    f_scale: float | None,
    phase_loss: bool,
    output_dir: str,
    bandwidth_hz: float,
):
    """Identify PX4 dynamics from ULog file.

    Models:
        First order: H(s) = exp(-tau*s) * K / (s + pole)
        Second order: H(s) = exp(-tau*s) * omega_n^2 / (s^2 + 2*zeta*omega_n*s + omega_n^2)

    Modes:
        rate: Angular velocity response (roll/pitch axes)
        accel: Thrust acceleration response (Z-axis acceleration)

    Examples:
        # Standard rate identification (second order)
        identify input.ulg

        # Acceleration mode with first order model
        identify input.ulg --mode accel --model-order 1

        # With robust loss
        identify input.ulg --mode accel --model-order 1 --robust-loss huber
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

    # Store plot data
    plot_data = {
        "ts": {},
        "u": {},
        "y": {},
    }
    results = {}

    if mode == "rate":
        # Get rate data
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

        input_unit = "rad/s"
        output_unit = "rad/s"

    else:  # mode == "accel"
        # Get acceleration data
        thrust_sp = get_thrust_acceleration_setpoint_data(ulog)
        accel_z = get_accel_z_data(ulog)

        if thrust_sp is None or accel_z is None:
            print("Error: Required topics not found")
            sys.exit(1)

        ts_acc, thrust_acc_sp = thrust_sp
        ts_accel, accel_z_meas = accel_z

        # For accel mode, process single axis
        axes_to_process = ["accel"]

        # Store data for processing
        roll_sp = thrust_acc_sp
        roll_meas = accel_z_meas
        pitch_sp = None
        pitch_meas = None

        ts_acc_for_roll = ts_acc
        ts_ang_for_roll = ts_accel

        input_unit = "m/s²"
        output_unit = "m/s²"

    # Process each axis
    for axis in axes_to_process:
        print(f"[{axis.upper()}]")

        # Select data based on mode
        if mode == "rate":
            if axis == "roll":
                u_raw = roll_sp
                y_raw = roll_meas
                ts_u_raw = ts_acc
                ts_y_raw = ts_ang
            else:
                u_raw = pitch_sp
                y_raw = pitch_meas
                ts_u_raw = ts_acc
                ts_y_raw = ts_ang
        else:  # accel mode
            u_raw = roll_sp
            y_raw = roll_meas
            ts_u_raw = ts_acc_for_roll
            ts_y_raw = ts_ang_for_roll

        # Extract offboard segment
        ts_u, u = extract_segment_data(ts_u_raw, u_raw, offboard_start, ident_duration)
        ts_y, y = extract_segment_data(ts_y_raw, y_raw, offboard_start, ident_duration)

        # Remove mean
        u = u - np.mean(u)
        y = y - np.mean(y)

        # Compute sample time from timestamps (in seconds)
        dt = float(np.median(np.diff(ts_u)))

        print(f"  Data points: {len(u)}")
        print(f"  Sample time: {dt:.4f} s ({1.0 / dt:.1f} Hz)")
        print(f"  Input std: {np.std(u):.4f} {input_unit}")
        print(f"  Output std: {np.std(y):.4f} {output_unit}")

        # Identify model with delay
        result = identify_axis(
            u,
            y,
            dt,
            max_delay_samples=max_delay,
            robust_loss=robust_loss,
            f_scale=f_scale,
            model_order=int(model_order),
            use_phase_loss=phase_loss,
        )

        if result is None:
            print(f"  Error: Identification failed")
            continue

        print(f"  Fit: {result.fit_pct:.1f}%")

        # Calculate and display separate magnitude/phase fit
        mag_fit, phase_fit = fit_magnitude_phase_percent(y, result.y_hat)
        print(f"  Magnitude fit: {mag_fit:.1f}%")
        print(f"  Phase fit: {phase_fit:.1f}%")

        print(f"  Delay: {result.delay_samples} samples ({result.tau:.3f} s)")

        if result.model_order == 1:
            print(f"  Model: First order")
            T = 1.0 / result.pole  # pole = 1/T
            print(f"  Time constant T: {T:.3f} s")
        else:
            print(f"  Model: Second order")
            print(
                f"  omega_n: {result.omega_n:.2f} rad/s ({result.omega_n / (2 * np.pi):.2f} Hz)"
            )
            print(f"  zeta: {result.zeta:.4f}")

        print(f"  Stable: {result.stable}")
        print(f"  Loss function: {result.loss_function}")
        if phase_loss:
            print(f"  Phase-only loss: True (optimizes phase matching)")
        if result.loss_function != "linear" and f_scale is not None:
            print(f"  f_scale: {f_scale:.3f} rad/s")
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
            "model_order": result.model_order,
            "gain": result.gain,
            "pole": result.pole,
            "omega_n": result.omega_n,
            "zeta": result.zeta,
            "tau": result.tau,
            "delay_samples": result.delay_samples,
            "poles": result.poles,
            "loss_function": result.loss_function,
            "dominant_freqs": fft_result,
        }

    # Generate plots
    if results:
        stem = ulg_path.stem
        # Add mode suffix to avoid overwriting files
        mode_suffix = f"_{mode}" if mode == "accel" else ""
        stem_with_mode = f"{stem}{mode_suffix}"

        # 1. Data filtering plot
        plot_data_filtering(
            stem_with_mode,
            plot_data["ts"],
            plot_data["u"],
            plot_data["y"],
            output_path / f"{stem_with_mode}_data_filtering.png",
        )
        print(f"  Data filtering plot saved")

        # 2. Fit plot
        plot_fit(stem_with_mode, results, output_path / f"{stem_with_mode}_fit.png")
        print(f"  Fit plot saved")

        # 3. Frequency analysis plot
        plot_frequency_analysis(
            stem_with_mode, results, output_path / f"{stem_with_mode}_bode.png", bandwidth_hz
        )
        print(f"  Frequency analysis plot saved")

        # 4. Final result plot
        plot_final_result(stem_with_mode, results, output_path / f"{stem_with_mode}_final.png")
        print(f"  Final result plot saved\n")

        # Save parameters
        params = {
            "ulg_file": str(ulg_path),
            "offboard_start": float(offboard_start),
            "offboard_duration": float(offboard_duration),
            "identification_duration": float(ident_duration),
            "mode": mode,
            "model_order": int(model_order),
            "identification_options": {
                "max_delay_samples": max_delay,
                "robust_loss": robust_loss,
                "f_scale": f_scale if f_scale is not None else "auto",
            },
            "axes": {},
        }

        for axis, data in results.items():
            freqs = data["dominant_freqs"]
            axis_params = {
                "sample_rate_hz": float(freqs["fs"]),
                "transfer_function": data["transfer_function"],
                "fit_percent": float(data["fit_pct"]),
                "model_order": data["model_order"],
                "tau_seconds": float(data["tau"]),
                "delay_samples": int(data["delay_samples"]),
                "poles": [complex(p) for p in data["poles"]],
                "loss_function": data["loss_function"],
                "dominant_freqs_input": [
                    {"freq_hz": f, "magnitude": m} for f, m in freqs["input"]
                ],
                "dominant_freqs_output": [
                    {"freq_hz": f, "magnitude": m} for f, m in freqs["output"]
                ],
            }

            # Add model-specific parameters
            if data["model_order"] == 1:
                T = 1.0 / data["pole"]  # pole = 1/T
                axis_params["time_constant_T"] = float(T)
                axis_params["pole"] = float(data["pole"])  # Keep pole for reference
                axis_params["gain"] = 1.0  # Fixed gain
            else:
                axis_params["omega_n"] = float(data["omega_n"])
                axis_params["omega_n_hz"] = float(data["omega_n"] / (2 * np.pi))
                axis_params["zeta"] = float(data["zeta"])

            params["axes"][axis] = axis_params

        # Update model string based on order
        if int(model_order) == 1:
            params["model"] = "H(s) = exp(-tau*s) / (T*s + 1)"
        else:
            params["model"] = "H(s) = exp(-tau*s) * omega_n^2 / (s^2 + 2*zeta*omega_n*s + omega_n^2)"

        params_file = output_path / f"{stem_with_mode}_params.yaml"
        with open(params_file, "w") as f:
            yaml.dump(params, f, default_flow_style=False, sort_keys=False)

        print(f"Parameters saved: {params_file}")

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
