#!/bin/bash

# 测试所有练习的参考答案

echo "======================================================================"
echo "  测试 Operator Ex3 所有练习"
echo "======================================================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 测试计数
TOTAL=0
PASSED=0
FAILED=0

# 测试函数
test_exercise() {
    local ex_num=$1
    local ex_name=$2
    
    echo "----------------------------------------------------------------------"
    echo "测试练习 $ex_num: $ex_name"
    echo "----------------------------------------------------------------------"
    
    TOTAL=$((TOTAL + 1))
    
    cd "solutions/ex$ex_num" || exit 1
    
    if go run . > /dev/null 2>&1; then
        echo -e "${GREEN}✓ 练习 $ex_num 测试通过${NC}"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}✗ 练习 $ex_num 测试失败${NC}"
        FAILED=$((FAILED + 1))
    fi
    
    cd ../.. || exit 1
    echo ""
}

# 测试框架
echo "----------------------------------------------------------------------"
echo "测试框架代码"
echo "----------------------------------------------------------------------"
TOTAL=$((TOTAL + 1))

cd framework || exit 1

if go test -v > /dev/null 2>&1; then
    echo -e "${GREEN}✓ 框架测试通过${NC}"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗ 框架测试失败${NC}"
    FAILED=$((FAILED + 1))
fi

cd .. || exit 1
echo ""

# 测试所有练习
test_exercise 1 "状态机与基础协调循环"
test_exercise 2 "资源创建与管理"
test_exercise 3 "更新与同步逻辑"
test_exercise 4 "删除与 Finalizer"
test_exercise 5 "错误处理与可观测性"

# 输出总结
echo "======================================================================"
echo "  测试总结"
echo "======================================================================"
echo "总计: $TOTAL"
echo -e "${GREEN}通过: $PASSED${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}失败: $FAILED${NC}"
else
    echo "失败: $FAILED"
fi
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 所有测试通过！${NC}"
    exit 0
else
    echo -e "${RED}❌ 有测试失败${NC}"
    exit 1
fi
