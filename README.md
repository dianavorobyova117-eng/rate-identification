# Rate Identification Tool

自动从PX4 ULog文件中辨识系统动力学。

## 功能

- 自动检测offboard模式时长
- **两种辨识模式**：
  - `rate`: 角速率环（roll/pitch轴）
  - `accel`: 推力加速度环（Z轴加速度）
- **两种模型阶数**：
  - 一阶模型：`H(s) = e^(-τ*s) * K / (s + pole)`
  - 二阶模型：`H(s) = e^(-τ*s) * ωₙ² / (s² + 2ζωₙs + ωₙ²)`
- **时延估计**：互相关粗估计 + 网格搜索精细优化
- **鲁棒拟合**：支持huber/cauchy等损失函数处理异常值
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

# 处理单个文件（二阶模型，角速率模式）
uv run scripts/identify.py input/your_file.ulg

# 推力加速度环辨识（推荐一阶模型）
uv run scripts/identify.py input/your_file.ulg --mode accel --model-order 1

# 指定辨识时长（秒）
uv run scripts/identify.py input/your_file.ulg --duration 3.0

# 只辨识特定轴
uv run scripts/identify.py input/your_file.ulg --axes pitch

# 使用鲁棒损失函数处理异常值
uv run scripts/identify.py input/your_file.ulg --mode accel --model-order 1 --robust-loss huber
```

## 参数说明

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--duration` | 辨识时长（秒） | 2.5 |
| `--mode` | 辨识模式：rate（角速率）或accel（加速度） | rate |
| `--model-order` | 模型阶数：1或2 | 2 |
| `--axes` | 辨识轴：roll/pitch/both | both |
| `--max-delay` | 最大时延搜索范围（采样点） | 30 |
| `--robust-loss` | 鲁棒损失：linear/huber/cauchy/soft_l1/arctan | linear |
| `--f-scale` | 鲁棒损失的软边界（rad/s或m/s²） | 自动检测 |

## 输出

每个ulg文件生成：
- `{filename}_data_filtering.png` - 归一化输入输出数据
- `{filename}_fit.png` - 时域拟合对比
- `{filename}_bode.png` - 频域分析（波特图）
- `{filename}_final.png` - 最终结果（输入、测量、拟合）
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

### 角速率环（二阶模型）
```
ROLL:
  Fit: 80.5%
  Delay: 1 sample
  Model: Second order
  omega_n: 45.2 rad/s (7.2 Hz)
  zeta: 0.65
  H(s) = exp(-0.020s) * 2043.0 / (s^2 + 58.8s + 2043.0)
```

### 推力加速度环（一阶模型）
```
ACCEL:
  Fit: 81.4%
  Delay: 7 samples (0.140 s)
  Model: First order
  Gain K: 9.868
  Pole: 17.626 rad/s
  H(s) = exp(-0.140s) * 9.868 / (s + 17.626)
```

## 算法说明

### 时延估计
采用两阶段方法：
1. **粗估计**：计算输入输出的互相关函数，找到最大相关对应的滞后
2. **精细搜索**：在粗估计±3采样点范围内网格搜索
3. **联合优化**：对每个时延值优化连续参数，选择残差最小的组合

### 时延实现方式
使用**一阶Pade近似**将时延项转换为有理函数：
```
e^(-τ*s) ≈ (1 - τ*s/2) / (1 + τ*s/2)
```

一阶系统使用标准惯性环节模型（K=1）：
```
H(s) = e^(-τ*s) / (T*s + 1)
```

使用Pade近似的完整形式：
```
H(s) ≈ (1 - τ*s/2) / (1 + τ*s/2) * 1 / (T*s + 1)
```

其中T为时间常数，K固定为1（标准形式）。

### 模型选择
- **角速率环**：推荐二阶模型，典型频率5-30Hz
- **加速度环**：推荐一阶模型（标准惯性环节），典型频率1-2πT

### 鲁棒拟合
当数据存在异常扰动时，使用鲁棒损失函数：
- `huber`：中等异常值，推荐默认
- `cauchy`：严重异常值，强抑制
