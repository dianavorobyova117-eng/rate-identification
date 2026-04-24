"""ULog data processing and offboard detection."""

from pathlib import Path

import numpy as np
import pyulog


def find_offboard_segment(ulog: pyulog.ULog) -> tuple[float, float] | None:
    """查找offboard模式的起始和结束时间.

    Returns:
        (start_time, end_time) 单位秒，如果未找到返回None
    """
    for d in ulog.data_list:
        if d.name == "vehicle_control_mode":
            offboard_mask = d.data["flag_control_offboard_enabled"] == 1
            offboard_indices = np.where(offboard_mask)[0]

            if len(offboard_indices) == 0:
                return None

            start_time = d.data["timestamp"][offboard_indices[0]] / 1e6
            end_time = d.data["timestamp"][offboard_indices[-1]] / 1e6
            return (start_time, end_time)

    return None


def get_acc_rates_data(ulog: pyulog.ULog) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """获取acc_rates_setpoint数据.

    Returns:
        (timestamp, rates_sp[0], rates_sp[1]) 或 None
    """
    for d in ulog.data_list:
        if d.name == "vehicle_acc_rates_setpoint":
            ts = d.data["timestamp"].astype(float) / 1e6
            roll_sp = d.data["rates_sp[0]"].astype(float)
            pitch_sp = d.data["rates_sp[1]"].astype(float)
            return (ts, roll_sp, pitch_sp)
    return None


def get_angular_velocity_data(ulog: pyulog.ULog) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """获取angular_velocity数据.

    Returns:
        (timestamp, xyz[0], xyz[1]) 或 None
    """
    for d in ulog.data_list:
        if d.name == "vehicle_angular_velocity":
            ts = d.data["timestamp"].astype(float) / 1e6
            roll = d.data["xyz[0]"].astype(float)
            pitch = d.data["xyz[1]"].astype(float)
            return (ts, roll, pitch)
    return None


def extract_segment_data(
    ts: np.ndarray,
    data: np.ndarray,
    start_time: float,
    duration: float,
    sample_rate: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """提取指定时间段的数据并重采样.

    Args:
        ts: 原始时间戳
        data: 原始数据
        start_time: 起始时间
        duration: 时长
        sample_rate: 目标采样率

    Returns:
        (ts_resampled, data_resampled)
    """
    end_time = start_time + duration

    # 创建目标时间轴
    ts_target = np.arange(start_time, end_time, 1.0 / sample_rate)

    # 重采样
    data_resampled = np.interp(ts_target, ts, data)

    return ts_target, data_resampled
