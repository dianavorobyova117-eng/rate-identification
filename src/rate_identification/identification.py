"""Core rate identification algorithms."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares


@dataclass
class FitResult:
    """结果数据类."""
    order: int
    theta: np.ndarray
    y_hat: np.ndarray
    poles: np.ndarray
    stable: bool
    fit_pct: float
    mse_val: float
    gain: float
    delay_samples: int
    transfer_function: str


def fit_percent(y: np.ndarray, y_hat: np.ndarray) -> float:
    """计算拟合百分比."""
    num = np.linalg.norm(y - y_hat)
    den = np.linalg.norm(y)
    if den < 1e-12:
        return 0.0
    return max(0.0, 100.0 * (1.0 - num / den))


def mse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """计算均方误差."""
    return float(np.mean((y - y_hat) ** 2))


def simulate_iir_with_delay(
    u: np.ndarray, theta: np.ndarray, delay: int, order: int, y_init: float | None = None
) -> np.ndarray:
    """模拟IIR系统带延迟."""
    a_coeffs = theta[:order]
    b_coeffs = theta[order:]

    y_hat = np.zeros(len(u), dtype=float)
    if y_init is not None:
        y_hat[: delay + order] = y_init

    for k in range(delay + order, len(u)):
        y_hat[k] = sum(-a_coeffs[i] * y_hat[k - 1 - i] for i in range(order))
        y_hat[k] += sum(b_coeffs[i] * u[k - delay - i] for i in range(order))
    return y_hat


def fit_iir_with_delay(
    u: np.ndarray, y: np.ndarray, order: int, max_delay: int = 20
) -> tuple[np.ndarray | None, int, np.ndarray | None, float]:
    """拟合IIR模型并搜索最佳延迟."""
    u = np.asarray(u, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    best_theta = None
    best_delay = 0
    best_y_hat = None
    best_fit = -1.0

    for delay in range(max_delay + 1):
        X_rows = []
        y_vals = []
        for k in range(delay + order, len(u)):
            row = [-y[k - 1 - i] for i in range(order)] + [
                u[k - delay - i] for i in range(order)
            ]
            X_rows.append(row)
            y_vals.append(y[k])

        if len(X_rows) < 2 * order:
            continue

        X = np.array(X_rows)
        y_vec = np.array(y_vals)

        theta, _, _, _ = np.linalg.lstsq(X, y_vec, rcond=None)

        a_coeffs = theta[:order]
        poles = np.roots([1.0] + list(a_coeffs))
        if not np.all(np.abs(poles) < 1.0):
            continue

        y_hat = simulate_iir_with_delay(u, theta, delay, order, y_init=y[0])
        valid_start = delay + order
        fit = fit_percent(y[valid_start:], y_hat[valid_start:])

        if fit > best_fit:
            best_fit = fit
            best_theta = theta
            best_delay = delay
            best_y_hat = y_hat

    return best_theta, best_delay, best_y_hat, best_fit


def format_transfer_function(theta: np.ndarray, order: int, delay: int) -> str:
    """格式化传递函数表达式."""
    a_coeffs = theta[:order]
    b_coeffs = theta[order:]

    num_terms = [f"{b:.6f} z^-{delay + i}" for i, b in enumerate(b_coeffs)]
    den_terms = ["1"] + [f"{a:.6f} z^-{i + 1}" for i, a in enumerate(a_coeffs)]

    return f"H(z) = z^-{delay} * ({' + '.join(num_terms)}) / ({' + '.join(den_terms)})"


def identify_axis(
    u: np.ndarray, y: np.ndarray, max_delay: int = 20, orders: list[int] = [1, 2]
) -> FitResult | None:
    """辨识单个轴的动力学."""
    best_result = None
    best_fit = -1.0

    for order in orders:
        theta, delay, y_hat, fit = fit_iir_with_delay(u, y, order, max_delay)
        if theta is not None:
            poles = np.roots([1.0] + list(theta[:order]))
            stable = bool(np.all(np.abs(poles) < 1.0))

            result = FitResult(
                order=order,
                theta=theta,
                y_hat=y_hat,
                poles=poles,
                stable=stable,
                fit_pct=fit,
                mse_val=mse(y, y_hat),
                gain=0.0,
                delay_samples=delay,
                transfer_function=format_transfer_function(theta, order, delay),
            )

            if fit > best_fit:
                best_fit = fit
                best_result = result

    return best_result
