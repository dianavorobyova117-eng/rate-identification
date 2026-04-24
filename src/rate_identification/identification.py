"""Core rate identification algorithms - continuous second-order model."""

from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.optimize import least_squares


@dataclass
class FitResult:
    """Identification result."""

    omega_n: float  # natural frequency [rad/s]
    zeta: float  # damping ratio [-]
    y_hat: np.ndarray
    poles: np.ndarray
    stable: bool
    fit_pct: float
    mse_val: float
    dt: float
    transfer_function: str


def fit_percent(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Calculate fit percentage."""
    num = np.linalg.norm(y - y_hat)
    den = np.linalg.norm(y)
    if den < 1e-12:
        return 0.0
    return max(0.0, 100.0 * (1.0 - num / den))


def mse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Calculate mean squared error."""
    return float(np.mean((y - y_hat) ** 2))


def simulate_second_order(
    u: np.ndarray, omega_n: float, zeta: float, dt: float, y_init: float = 0.0
) -> np.ndarray:
    """Simulate continuous second-order system via ZOH discretization.

    H(s) = omega_n^2 / (s^2 + 2*zeta*omega_n*s + omega_n^2)

    Args:
        u: Input signal
        omega_n: Natural frequency [rad/s]
        zeta: Damping ratio [-]
        dt: Sample time [s]
        y_init: Initial output value

    Returns:
        Simulated output y_hat
    """
    # Continuous transfer function
    num_s = [omega_n**2]
    den_s = [1.0, 2.0 * zeta * omega_n, omega_n**2]

    # Discretize using ZOH
    num_z, den_z, _ = signal.cont2discrete((num_s, den_s), dt, method="zoh")

    # Simulate discrete system
    num_z = num_z.flatten()
    den_z = den_z.flatten()

    y_hat = np.zeros(len(u), dtype=float)
    order = len(den_z) - 1

    # Set initial conditions
    y_hat[:order] = y_init

    # IIR filter simulation
    for k in range(order, len(u)):
        y_hat[k] = -sum(den_z[i + 1] * y_hat[k - 1 - i] for i in range(order))
        y_hat[k] += sum(num_z[i] * u[k - i] for i in range(min(len(num_z), k + 1)))

    return y_hat


def residuals(
    params: np.ndarray, u: np.ndarray, y: np.ndarray, dt: float
) -> np.ndarray:
    """Compute residuals for least-squares fitting."""
    omega_n, zeta = params
    y_hat = simulate_second_order(u, omega_n, zeta, dt, y_init=y[0])
    return y - y_hat


def fit_second_order(
    u: np.ndarray, y: np.ndarray, dt: float
) -> tuple[float | None, float | None, np.ndarray | None, float]:
    """Fit continuous second-order model H(s) = omega_n^2 / (s^2 + 2*zeta*omega_n*s + omega_n^2).

    Args:
        u: Input signal
        y: Measured output
        dt: Sample time [s]

    Returns:
        (omega_n, zeta, y_hat, fit_pct) or (None, None, None, -1.0) if failed
    """
    u = np.asarray(u, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    # Initial guess: typical rate loop
    omega_n0 = 20.0  # [rad/s]
    zeta0 = 0.7  # [-]

    x0 = np.array([omega_n0, zeta0])

    # Bounds: omega_n > 0, 0 < zeta < 2 (underdamped to overdamped)
    lower = np.array([1.0, 0.01])
    upper = np.array([200.0, 2.0])

    try:
        result = least_squares(
            residuals,
            x0,
            args=(u, y, dt),
            bounds=(lower, upper),
            method="trf",
            max_nfev=200,
        )

        if not result.success:
            return None, None, None, -1.0

        omega_n, zeta = result.x
        y_hat = simulate_second_order(u, omega_n, zeta, dt, y_init=y[0])
        fit = fit_percent(y, y_hat)

        return omega_n, zeta, y_hat, fit

    except Exception:
        return None, None, None, -1.0


def format_continuous_tf(omega_n: float, zeta: float) -> str:
    """Format continuous transfer function string."""
    return f"H(s) = {omega_n**2:.2f} / (s^2 + {2 * zeta * omega_n:.2f}s + {omega_n**2:.2f})"


def identify_axis(u: np.ndarray, y: np.ndarray, dt: float) -> FitResult | None:
    """Identify second-order continuous dynamics for one axis.

    Args:
        u: Input signal
        y: Measured output
        dt: Sample time [s]

    Returns:
        FitResult or None if identification failed
    """
    omega_n, zeta, y_hat, fit = fit_second_order(u, y, dt)
    if omega_n is None:
        return None

    # Check stability: poles of s^2 + 2*zeta*omega_n*s + omega_n^2 = 0
    # For stable system: omega_n > 0 and zeta > 0
    stable = bool(omega_n > 0 and zeta > 0)

    # Continuous poles
    den_s = [1.0, 2.0 * zeta * omega_n, omega_n**2]
    poles = np.roots(den_s)

    return FitResult(
        omega_n=omega_n,
        zeta=zeta,
        y_hat=y_hat,
        poles=poles,
        stable=stable,
        fit_pct=fit,
        mse_val=mse(y, y_hat),
        dt=dt,
        transfer_function=format_continuous_tf(omega_n, zeta),
    )
