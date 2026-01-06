#!/bin/bash
# 运行测试脚本
# 用法: ./scripts/run_tests.sh [options]
#   -a, --all       运行所有测试，包括慢速测试
#   -c, --coverage  运行带覆盖率报告的测试
#   -v, --verbose   详细输出
#   -h, --help      显示帮助信息

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 默认选项
RUN_ALL=false
WITH_COVERAGE=false
VERBOSE=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--all)
            RUN_ALL=true
            shift
            ;;
        -c|--coverage)
            WITH_COVERAGE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "用法: $0 [options]"
            echo ""
            echo "选项:"
            echo "  -a, --all       运行所有测试，包括慢速测试"
            echo "  -c, --coverage  运行带覆盖率报告的测试"
            echo "  -v, --verbose   详细输出"
            echo "  -h, --help      显示帮助信息"
            exit 0
            ;;
        *)
            echo -e "${RED}未知选项: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}    智能枪械训练系统 - 单元测试${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: 未找到python3${NC}"
    exit 1
fi

# 检查pytest
if ! python3 -m pytest --version &> /dev/null; then
    echo -e "${YELLOW}安装pytest...${NC}"
    pip install pytest pytest-cov pytest-mock
fi

# 构建pytest命令
PYTEST_CMD="python3 -m pytest"

if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -v"
fi

if [ "$RUN_ALL" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --runslow"
fi

if [ "$WITH_COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=backend --cov-report=term-missing --cov-report=html"
fi

# 运行测试
echo -e "${YELLOW}运行命令: $PYTEST_CMD${NC}"
echo ""

cd "$(dirname "$0")/.."
$PYTEST_CMD

# 测试完成
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}    测试完成!${NC}"
echo -e "${GREEN}========================================${NC}"

if [ "$WITH_COVERAGE" = true ]; then
    echo ""
    echo -e "${YELLOW}覆盖率报告已生成: htmlcov/index.html${NC}"
fi

