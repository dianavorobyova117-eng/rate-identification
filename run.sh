#!/bin/bash
# 一行命令处理所有ulg文件

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 创建输入目录（如果不存在）
mkdir -p input output

# 查找所有ulg文件
ULG_FILES=(input/*.ulg)

if [ ${#ULG_FILES[@]} -eq 0 ] || [ ! -f "${ULG_FILES[0]}" ]; then
    echo "No .ulg files found in input/ directory"
    echo "Please copy your ULog files to input/"
    exit 1
fi

echo "Found ${#ULG_FILES[@]} ULog file(s)"
echo ""

# 处理每个文件
for ulg_file in "${ULG_FILES[@]}"; do
    echo "Processing: $ulg_file"
    uv run scripts/identify.py "$ulg_file"
    echo ""
done

echo "All files processed!"
echo "Results saved to output/"
