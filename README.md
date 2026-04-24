# Rate Identification Tool

自动从PX4 ULog文件中辨识roll和pitch的rate dynamics。

## 功能

- 自动检测offboard模式时长
- 使用vehicle_acc_rates_setpoint作为输入，vehicle_angular_velocity作为输出
- 同时辨识roll和pitch两个轴
- 生成传递函数表达式和丰富的可视化结果

## 安装

使用uv管理环境：

```bash
cd /home/ivan/jetson_sync/rate_identification
uv sync
```

## 快速开始

```bash
# 一行命令处理所有ulg文件
./run.sh
```

## 单个文件处理

```bash
# 基本使用（自动检测offboard时长）
uv run scripts/identify.py input/your_file.ulg

# 指定辨识时长（秒）
uv run scripts/identify.py input/your_file.ulg --duration 3.0

# 只辨识特定轴
uv run scripts/identify.py input/your_file.ulg --axes pitch
```

## 输出

每个ulg文件生成：
- `{filename}_data_filtering.png` - 归一化输入输出数据
- `{filename}_fit.png` - 时域拟合对比（多阶模型）
- `{filename}_bode.png` - 频域分析（波特图）
- `{filename}_params.yaml` - 辨识参数和传递函数
