# Rate Identification Tool

自动从PX4 ULog文件中辨识roll和pitch的rate dynamics。

## 功能

- 自动检测offboard模式时长
- 使用vehicle_acc_rates_setpoint作为输入，vehicle_angular_velocity作为输出
- 同时辨识roll和pitch两个轴
- 生成传递函数表达式和丰富的可视化结果

## 快速开始

### 1. 安装依赖

```bash
# 克隆仓库
git clone https://github.com/dianavorobyova117-eng/rate-identification.git
cd rate-identification

# 运行setup脚本（自动检测用户名并创建环境）
./setup.sh
```

### 2. 使用

```bash
# 一行命令处理所有ulg文件
./run.sh

# 或处理单个文件
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

## 目录结构

```
rate-identification/
├── input/           # 放置你的.ulg文件
├── output/          # 辨识结果输出
├── run.sh           # 一行命令执行
├── setup.sh         # 环境设置脚本
└── src/rate_identification/
```

## 示例结果

```
ROLL:
  Fit: 80.5%
  Delay: 1 sample
  H(z) = z^-1 * (0.351z^-1 + 0.293z^-2) / (1 - 0.458z^-1 + 0.057z^-2)

PITCH:
  Fit: 82.5%
  Delay: 0 samples
  H(z) = (-0.178 + 0.633z^-1) / (1 - 0.734z^-1 + 0.219z^-2)
```
