# §2.6 第 K 小/大

通过二分答案来找第 K 小/大的元素。

## 核心思想

- **第 k 小** = 求最小的 x，满足 ≤x 的数至少有 k 个
- **第 k 大** = 求最大的 x，满足 ≥x 的数至少有 k 个

## 理解方式

### 第 K 小

例如数组 `[1,1,1,2,2]`：
- 第 1 小、第 2 小、第 3 小的数都是 1
- 第 4 小和第 5 小的数都是 2

求第 k 小就是找最小的 x，使得 ≤x 的数有至少 k 个。

### 第 K 大

类似地，第 k 大就是找最大的 x，使得 ≥x 的数有至少 k 个。

## 模板

### 第 K 小（求最小）

```python
def kthSmallest(...):
    def check(x):
        # 统计 <= x 的数有多少个
        count = count_less_or_equal(x)
        return count >= k
    
    # 二分 x（求最小）
    left = 最小可能值 - 1
    right = 最大可能值
    
    while left + 1 < right:
        mid = (left + right) // 2
        if check(mid):  # <= mid 的数足够多
            right = mid  # 尝试更小的
        else:
            left = mid
    
    return right
```

### 第 K 大（求最大）

```python
def kthLargest(...):
    def check(x):
        # 统计 >= x 的数有多少个
        count = count_greater_or_equal(x)
        return count >= k
    
    # 二分 x（求最大）
    left = 最小可能值
    right = 最大可能值 + 1
    
    while left + 1 < right:
        mid = (left + right) // 2
        if check(mid):  # >= mid 的数足够多
            left = mid  # 尝试更大的
        else:
            right = mid
    
    return left
```

## 注意事项

1. **k 从 1 开始**：一般规定 k 从 1 开始，而不是从 0 开始
2. **可以用堆**：部分题目也可以用堆（优先队列）解决
3. **统计个数**：关键是如何高效地统计 ≤x 或 ≥x 的个数

## 适用场景

- 数据量大，不适合排序
- 数据分布有规律（如乘法表、有序矩阵）
- 需要在线处理（流式数据）

---

**返回**: [二分答案](../README.md) | [二分查找专题](../../README.md)
