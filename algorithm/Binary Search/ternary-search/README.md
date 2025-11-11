# 三、三分法

三分法用于在单峰函数（先增后减或先减后增）中寻找极值点。

## 核心思想

对于单峰函数：
- 二分法找单调函数的零点
- **三分法找单峰函数的极值点**

## 适用场景

函数满足以下特征之一：
1. **单峰**：先增后减（找最大值）
2. **单谷**：先减后增（找最小值）
3. **凸函数**或**凹函数**

## 算法原理

将区间三等分，取两个分点 `m1` 和 `m2`：
- 如果 `f(m1) < f(m2)`：极值在 `[m1, right]` 中
- 如果 `f(m1) > f(m2)`：极值在 `[left, m2]` 中
- 如果 `f(m1) == f(m2)`：极值在 `[m1, m2]` 中

## 模板

### 求最大值（先增后减）

```python
def ternary_search_max(left, right, eps=1e-6):
    """
    在 [left, right] 中找单峰函数的最大值
    eps: 精度
    """
    while right - left > eps:
        m1 = left + (right - left) / 3
        m2 = right - (right - left) / 3
        
        if f(m1) < f(m2):
            left = m1  # 最大值在右边
        else:
            right = m2  # 最大值在左边
    
    return (left + right) / 2
```

### 求最小值（先减后增）

```python
def ternary_search_min(left, right, eps=1e-6):
    """
    在 [left, right] 中找单谷函数的最小值
    """
    while right - left > eps:
        m1 = left + (right - left) / 3
        m2 = right - (right - left) / 3
        
        if f(m1) > f(m2):
            left = m1  # 最小值在右边
        else:
            right = m2  # 最小值在左边
    
    return (left + right) / 2
```

## 整数版本

如果定义域是整数：

```python
def ternary_search_int(left, right):
    """整数三分法"""
    while left + 2 < right:
        m1 = left + (right - left) // 3
        m2 = right - (right - left) // 3
        
        if f(m1) < f(m2):
            left = m1
        else:
            right = m2
    
    # 最后检查 left, left+1, right
    best = left
    for x in range(left, right + 1):
        if f(x) > f(best):  # 求最大值
            best = x
    return best
```

## 与二分法的对比

| 特征 | 二分法 | 三分法 |
|------|--------|--------|
| 适用场景 | 单调函数 | 单峰函数 |
| 目标 | 找零点/边界 | 找极值点 |
| 复杂度 | O(log n) | O(log n) |
| 每次缩小 | 1/2 | 2/3 |

## 注意事项

1. **函数必须单峰**：如果有多个峰，三分法可能找不到全局最优
2. **精度问题**：浮点三分需要设置合适的 eps
3. **整数问题**：最后需要检查几个候选点
4. **可以用黄金分割**：比例约为 0.618，理论上更优

## 实际应用

在 LeetCode 中，三分法题目较少，很多可以用其他方法解决。

---

**返回**: [二分查找专题](../README.md)
