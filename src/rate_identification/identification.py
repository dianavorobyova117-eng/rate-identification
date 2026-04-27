"""Core rate identification algorithms - continuous second-order model."""

from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.optimize import least_squares


@dataclass
class FitResult:
    """Identification result."""

    omega_n: float  # natural frequency [rad/s] (for second order)
    zeta: float  # damping ratio [-] (for second order)
    gain: float  # gain K (for first order)
    pole: float  # pole location [rad/s] (for first order)
    tau: float  # time delay [s]
    delay_samples: int  # delay in samples
    y_hat: np.ndarray
    poles: np.ndarray
    stable: bool
    fit_pct: float
    mse_val: float
    dt: float
    transfer_function: str
    model_order: int  # 1 for first order, 2 for second order
    loss_function: str = "linear"  # loss function used


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


def simulate_first_order(
    u: np.ndarray, gain: float, pole: float, dt: float, y_init: float = 0.0
) -> np.ndarray:
    """Simulate continuous first-order system via ZOH discretization.

    H(s) = K / (s + pole)

    Args:
        u: Input signal
        gain: Gain K [-]
        pole: Pole location [rad/s]
        dt: Sample time [s]
        y_init: Initial output value

    Returns:
        Simulated output y_hat
    """
    # Continuous transfer function: K / (s + pole)
    num_s = [gain]
    den_s = [1.0, pole]

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


def format_continuous_tf_with_delay(omega_n: float, zeta: float, tau: float) -> str:
    """Format continuous transfer function string with delay."""
    gain = omega_n**2
    term1 = f"{gain:.2f}"
    term2 = f"{2 * zeta * omega_n:.2f}"
    term3 = f"{gain:.2f}"

    if tau < 0.001:
        return f"H(s) = {term1} / (s^2 + {term2}s + {term3})"
    else:
        return f"H(s) = exp(-{tau:.3f}s) * {term1} / (s^2 + {term2}s + {term3})"


def format_first_order_tf(T: float) -> str:
    """Format first-order transfer function string (standard form)."""
    return f"H(s) = 1 / ({T:.3f}*s + 1)"


def format_first_order_tf_with_delay(T: float, tau: float) -> str:
    """Format first-order transfer function string with delay using Pade approximation."""
    if tau < 0.001:
        return f"H(s) = 1 / ({T:.3f}*s + 1)"
    else:
        # Show Pade approximation form
        return f"H(s) ≈ (1 - {tau:.3f}s/2) / (1 + {tau:.3f}s/2) * 1 / ({T:.3f}*s + 1)"


def simulate_first_order_integral(
    u: np.ndarray, T: float, dt: float, y_init: float = 0.0
) -> np.ndarray:
    """Simulate continuous first-order integral system via ZOH discretization.

    H(s) = 1 / (T*s + 1)

    Args:
        u: Input signal
        T: Time constant [s]
        dt: Sample time [s]
        y_init: Initial output value

    Returns:
        Simulated output y_hat
    """
    # Continuous transfer function: 1 / (T*s + 1) = 1 / (T*s + 1)
    num_s = [1.0]
    den_s = [T, 1.0]

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


def residuals_first_order(
    params: np.ndarray, u: np.ndarray, y: np.ndarray, dt: float
) -> np.ndarray:
    """Compute residuals for first-order least-squares fitting."""
    T = params[0]
    y_hat = simulate_first_order_integral(u, T, dt, y_init=y[0])
    return y - y_hat


def fit_first_order(
    u: np.ndarray, y: np.ndarray, dt: float
) -> tuple[float | None, np.ndarray | None, float]:
    """Fit continuous first-order model H(s) = 1 / (T*s + 1).

    Args:
        u: Input signal
        y: Measured output
        dt: Sample time [s]

    Returns:
        (T, y_hat, fit_pct) or (None, None, -1.0) if failed
    """
    u = np.asarray(u, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    # Initial guess: T ~ 0.1s (typical time constant)
    T0 = 0.1

    x0 = np.array([T0])

    # Bounds: T > 0.001, T < 10
    lower = np.array([0.001])
    upper = np.array([10.0])

    try:
        result = least_squares(
            residuals_first_order,
            x0,
            args=(u, y, dt),
            bounds=(lower, upper),
            method="trf",
            max_nfev=200,
        )

        if not result.success:
            return None, None, -1.0

        T = result.x[0]
        y_hat = simulate_first_order_integral(u, T, dt, y_init=y[0])
        fit = fit_percent(y, y_hat)

        return T, y_hat, fit

    except Exception:
        return None, None, -1.0


def estimate_delay_cross_correlation(
    u: np.ndarray, y: np.ndarray, max_delay_samples: int = 30
) -> int:
    """Estimate delay using cross-correlation (coarse estimate).

    Computes cross-correlation between input and output, finding the lag
    that maximizes correlation.

    Args:
        u: Input signal
        y: Output signal
        max_delay_samples: Maximum delay to search [samples]

    Returns:
        Estimated delay in samples
    """
    u_norm = u - np.mean(u)
    y_norm = y - np.mean(y)

    # Compute cross-correlation
    correlation = np.correlate(y_norm, u_norm, mode="valid")

    # correlation length is len(u_norm) - len(u_norm) + 1 = 1
    # For mode='valid', we need to use mode='full' to get lags
    correlation = np.correlate(y_norm, u_norm, mode="full")

    import scipy.signal as signal

    lags = signal.correlation_lags(len(u_norm), len(u_norm), mode="full")

    # Find lag that maximizes correlation (positive lag means y lags u)
    valid_mask = (lags >= 0) & (lags <= max_delay_samples)
    if not np.any(valid_mask):
        return 0

    valid_correlation = correlation[valid_mask]
    valid_lags = lags[valid_mask]

    if len(valid_lags) == 0:
        return 0

    best_lag = valid_lags[np.argmax(valid_correlation)]

    return int(best_lag)


def simulate_second_order_with_delay(
    u: np.ndarray,
    omega_n: float,
    zeta: float,
    tau: float,
    dt: float,
    y_init: float = 0.0,
) -> np.ndarray:
    """Simulate continuous second-order system with delay via ZOH discretization.

    H(s) = exp(-tau*s) * omega_n^2 / (s^2 + 2*zeta*omega_n*s + omega_n^2)

    The delay is implemented as integer samples: delay_samples = round(tau / dt)

    Args:
        u: Input signal
        omega_n: Natural frequency [rad/s]
        zeta: Damping ratio [-]
        tau: Time delay [s]
        dt: Sample time [s]
        y_init: Initial output value

    Returns:
        Simulated output y_hat
    """
    # Simulate undelayed system
    y_undelayed = simulate_second_order(u, omega_n, zeta, dt, y_init)

    # Apply delay (integer samples)
    delay_samples = int(round(tau / dt))

    # Shift output by delay_samples (fill beginning with y_init)
    y_hat = np.full_like(u, y_init)
    if delay_samples < len(u):
        if delay_samples > 0:
            y_hat[delay_samples:] = y_undelayed[:-delay_samples]
        else:
            y_hat = y_undelayed

    return y_hat


def simulate_first_order_with_delay(
    u: np.ndarray,
    T: float,
    tau: float,
    dt: float,
    y_init: float = 0.0,
) -> np.ndarray:
    """Simulate continuous first-order system with delay using Pade approximation.

    H(s) = exp(-tau*s) / (T*s + 1)

    Using first-order Pade approximation: exp(-tau*s) ≈ (1 - tau*s/2) / (1 + tau*s/2)

    Combined transfer function:
    H(s) = (1 - tau*s/2) / [(T*s + 1) * (1 + tau*s/2)]

    Args:
        u: Input signal
        T: Time constant [s]
        tau: Time delay [s]
        dt: Sample time [s]
        y_init: Initial output value

    Returns:
        Simulated output y_hat
    """
    if tau < 0.001:
        # Negligible delay, use undelayed system
        return simulate_first_order_integral(u, T, dt, y_init)

    # First-order Pade approximation: exp(-tau*s) ≈ (1 - tau*s/2) / (1 + tau*s/2)
    # H(s) = (1 - tau*s/2) / [(T*s + 1) * (1 + tau*s/2)]
    #
    # Expand denominator: (T*s + 1) * (1 + tau*s/2)
    #                   = T*s + 1 + T*tau*s²/2 + tau*s/2
    #                   = (T*tau/2)*s² + (T + tau/2)*s + 1
    #
    # Expand numerator: 1 - tau*s/2
    #
    # Standard form: num_s = [a1, a0] for a1*s + a0
    #                 den_s = [b2, b1, b0] for b2*s² + b1*s + b0

    num_s = [-tau / 2, 1]
    den_s = [T * tau / 2, T + tau / 2, 1]

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


def residuals_with_delay(
    params: np.ndarray,
    u: np.ndarray,
    y: np.ndarray,
    dt: float,
    delay_samples: int,
) -> np.ndarray:
    """Compute residuals for least-squares fitting with discrete delay.

    Args:
        params: [omega_n, zeta]
        u: Input signal
        y: Measured output
        dt: Sample time [s]
        delay_samples: Delay in samples (integer)

    Returns:
        Residuals (valid portion only, excluding delay transient)
    """
    omega_n, zeta = params
    tau = delay_samples * dt

    y_hat = simulate_second_order_with_delay(u, omega_n, zeta, tau, dt, y_init=y[0])

    # Exclude transient period (delay + 5 samples for settling)
    valid_start = min(delay_samples + 5, len(y))

    return y[valid_start:] - y_hat[valid_start:]


def fit_second_order_with_delay(
    u: np.ndarray,
    y: np.ndarray,
    dt: float,
    max_delay_samples: int = 30,
    loss: str = "linear",
    f_scale: float = 1.0,
) -> tuple[float | None, float | None, float | None, int | None, np.ndarray | None, float]:
    """Fit second-order model with delay: H(s) = exp(-tau*s) * omega_n^2 / (s^2 + 2*zeta*omega_n*s + omega_n^2).

    Uses two-stage approach:
    1. Grid search over discrete delay values
    2. Continuous optimization of (omega_n, zeta) for each delay

    Args:
        u: Input signal
        y: Measured output
        dt: Sample time [s]
        max_delay_samples: Maximum delay to search [samples]
        loss: Loss function ('linear', 'huber', 'cauchy', 'soft_l1', 'arctan')
        f_scale: Soft margin between inlier and outlier residuals

    Returns:
        (omega_n, zeta, tau, delay_samples, y_hat, fit_pct) or (None,...) if failed
    """
    u = np.asarray(u, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    # Stage 1: Coarse delay estimation via cross-correlation
    coarse_delay = estimate_delay_cross_correlation(u, y, max_delay_samples)

    # Stage 2: Grid search around coarse estimate (±3 samples)
    search_delays = np.arange(
        max(0, coarse_delay - 3), min(max_delay_samples + 1, coarse_delay + 4)
    )

    best_result = None
    best_cost = np.inf

    # Initial guess for continuous parameters
    omega_n0 = 20.0
    zeta0 = 0.7
    x0 = np.array([omega_n0, zeta0])

    # Bounds
    lower = np.array([1.0, 0.01])
    upper = np.array([200.0, 2.0])

    for delay_samples in search_delays:
        try:
            result = least_squares(
                lambda p: residuals_with_delay(p, u, y, dt, int(delay_samples)),
                x0,
                bounds=(lower, upper),
                method="trf",
                max_nfev=200,
                loss=loss,
                f_scale=f_scale,
            )

            if not result.success:
                continue

            # Compute cost for comparison
            cost = 0.5 * np.sum(result.fun**2)

            if cost < best_cost:
                best_cost = cost
                omega_n, zeta = result.x
                tau = delay_samples * dt
                y_hat = simulate_second_order_with_delay(
                    u, omega_n, zeta, tau, dt, y_init=y[0]
                )
                fit = fit_percent(y, y_hat)
                best_result = (omega_n, zeta, tau, int(delay_samples), y_hat, fit)

        except Exception:
            continue

    if best_result is None:
        return None, None, None, None, None, -1.0

    return best_result


def residuals_first_order_with_delay(
    params: np.ndarray,
    u: np.ndarray,
    y: np.ndarray,
    dt: float,
    delay_samples: int,
) -> np.ndarray:
    """Compute residuals for first-order least-squares fitting with discrete delay.

    Args:
        params: [T] - time constant
        u: Input signal
        y: Measured output
        dt: Sample time [s]
        delay_samples: Delay in samples (integer)

    Returns:
        Residuals (valid portion only, excluding delay transient)
    """
    T = params[0]
    tau = delay_samples * dt

    y_hat = simulate_first_order_with_delay(u, T, tau, dt, y_init=y[0])

    # Exclude transient period (delay + 5 samples for settling)
    valid_start = min(delay_samples + 5, len(y))

    return y[valid_start:] - y_hat[valid_start:]


def fit_first_order_with_delay(
    u: np.ndarray,
    y: np.ndarray,
    dt: float,
    max_delay_samples: int = 30,
    loss: str = "linear",
    f_scale: float = 1.0,
) -> tuple[float | None, float | None, int | None, np.ndarray | None, float]:
    """Fit first-order model with delay: H(s) = exp(-tau*s) / (T*s + 1).

    Uses two-stage approach:
    1. Grid search over discrete delay values
    2. Continuous optimization of T for each delay

    Args:
        u: Input signal
        y: Measured output
        dt: Sample time [s]
        max_delay_samples: Maximum delay to search [samples]
        loss: Loss function ('linear', 'huber', 'cauchy', 'soft_l1', 'arctan')
        f_scale: Soft margin between inlier and outlier residuals

    Returns:
        (T, tau, delay_samples, y_hat, fit_pct) or (None,...) if failed
    """
    u = np.asarray(u, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    # Stage 1: Coarse delay estimation via cross-correlation
    coarse_delay = estimate_delay_cross_correlation(u, y, max_delay_samples)

    # Stage 2: Grid search around coarse estimate (±3 samples)
    search_delays = np.arange(
        max(0, coarse_delay - 3), min(max_delay_samples + 1, coarse_delay + 4)
    )

    best_result = None
    best_cost = np.inf

    # Initial guess for continuous parameters
    T0 = 0.1  # Initial time constant [s]
    x0 = np.array([T0])

    # Bounds: T > 0.001, T < 10
    lower = np.array([0.001])
    upper = np.array([10.0])

    for delay_samples in search_delays:
        try:
            result = least_squares(
                lambda p: residuals_first_order_with_delay(p, u, y, dt, int(delay_samples)),
                x0,
                bounds=(lower, upper),
                method="trf",
                max_nfev=200,
                loss=loss,
                f_scale=f_scale,
            )

            if not result.success:
                continue

            # Compute cost for comparison
            cost = 0.5 * np.sum(result.fun**2)

            if cost < best_cost:
                best_cost = cost
                T = result.x[0]
                tau = delay_samples * dt
                y_hat = simulate_first_order_with_delay(
                    u, T, tau, dt, y_init=y[0]
                )
                fit = fit_percent(y, y_hat)
                best_result = (T, tau, int(delay_samples), y_hat, fit)

        except Exception:
            continue

    if best_result is None:
        return None, None, None, None, -1.0

    return best_result


def identify_axis(
    u: np.ndarray,
    y: np.ndarray,
    dt: float,
    max_delay_samples: int = 30,
    robust_loss: str = "linear",
    f_scale: float | None = None,
    model_order: int = 2,
) -> FitResult | None:
    """Identify continuous dynamics with delay for one axis.

    Models:
        - First order: H(s) = exp(-tau*s) / (T*s + 1)
        - Second order: H(s) = exp(-tau*s) * omega_n^2 / (s^2 + 2*zeta*omega_n*s + omega_n^2)

    Args:
        u: Input signal
        y: Measured output
        dt: Sample time [s]
        max_delay_samples: Maximum delay to search [samples] (default: 30)
        robust_loss: Robust loss function ('linear', 'huber', 'cauchy', 'soft_l1', 'arctan')
        f_scale: Soft margin for robust loss (None = auto-detect from signal std)
        model_order: Model order (1 or 2, default: 2)

    Returns:
        FitResult or None if identification failed
    """
    # Auto-detect f_scale if not specified
    if f_scale is None and robust_loss != "linear":
        f_scale = 1.5 * np.std(y)

    if model_order == 1:
        # Fit first-order model with delay
        result = fit_first_order_with_delay(
            u,
            y,
            dt,
            max_delay_samples=max_delay_samples,
            loss=robust_loss,
            f_scale=f_scale if f_scale is not None else 1.0,
        )

        if result[0] is None:
            return None

        T, tau, delay_samples, y_hat, fit = result

        # Check stability: T > 0
        stable = bool(T > 0)

        # Continuous pole: s = -1/T
        poles = np.array([-1.0 / T])

        return FitResult(
            omega_n=0.0,
            zeta=0.0,
            gain=1.0,
            pole=1.0 / T,
            tau=tau,
            delay_samples=delay_samples,
            y_hat=y_hat,
            poles=poles,
            stable=stable,
            fit_pct=fit,
            mse_val=mse(y, y_hat),
            dt=dt,
            transfer_function=format_first_order_tf_with_delay(T, tau),
            model_order=1,
            loss_function=robust_loss,
        )
    else:
        # Fit second-order model with delay
        result = fit_second_order_with_delay(
            u,
            y,
            dt,
            max_delay_samples=max_delay_samples,
            loss=robust_loss,
            f_scale=f_scale if f_scale is not None else 1.0,
        )

        if result[0] is None:
            return None

        omega_n, zeta, tau, delay_samples, y_hat, fit = result

        # Check stability: omega_n > 0 and zeta > 0
        stable = bool(omega_n > 0 and zeta > 0)

        # Continuous poles (delay doesn't affect poles, only adds phase lag)
        den_s = [1.0, 2.0 * zeta * omega_n, omega_n**2]
        poles = np.roots(den_s)

        return FitResult(
            omega_n=omega_n,
            zeta=zeta,
            gain=0.0,
            pole=0.0,
            tau=tau,
            delay_samples=delay_samples,
            y_hat=y_hat,
            poles=poles,
            stable=stable,
            fit_pct=fit,
            mse_val=mse(y, y_hat),
            dt=dt,
            transfer_function=format_continuous_tf_with_delay(omega_n, zeta, tau),
            model_order=2,
            loss_function=robust_loss,
        )
