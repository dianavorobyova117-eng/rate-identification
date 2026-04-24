#!/bin/bash
# 环境设置脚本 - 自动检测用户名并创建虚拟环境

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================"
echo "Rate Identification Tool - Setup"
echo "======================================"
echo ""

# 检查uv是否安装
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed"
    echo "Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✓ uv found: $(uv --version)"

# 创建必要的目录
echo ""
echo "Creating directories..."
mkdir -p input output
echo "✓ Directories created: input/ output/"

# 同步依赖
echo ""
echo "Installing dependencies..."
uv sync
echo "✓ Dependencies installed"

# 检查示例文件
echo ""
if [ -f "input/w1d4_frametest3_pitch_p-.ulg" ]; then
    echo "✓ Example ULog file found in input/"
    echo ""
    echo "You can run the example now:"
    echo "  ./run.sh"
else
    echo "Note: No ULog files in input/ directory"
    echo ""
    echo "To use this tool:"
    echo "  1. Copy your .ulg files to input/"
    echo "  2. Run: ./run.sh"
fi

echo ""
echo "======================================"
echo "Setup complete!"
echo "======================================"
