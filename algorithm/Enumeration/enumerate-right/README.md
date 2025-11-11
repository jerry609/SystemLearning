# §0.1 枚举右，维护左

## 📖 核心思想

对于**双变量问题**，例如两数之和 $a_i + a_j = t$，可以：
1. **枚举右边**的 $a_j$
2. 转换成**单变量问题**：在 $a_j$ 左边查找是否有 $a_i = t - a_j$
3. 用**哈希表维护**左侧已经遍历过的元素

这个技巧叫做：**枚举右，维护左**。

## 💡 为什么这样做？

### 传统做法（暴力枚举）
```python
# 时间复杂度 O(n²)
for i in range(n):
    for j in range(i+1, n):
        if nums[i] + nums[j] == target:
            return [i, j]
```

### 优化做法（枚举右，维护左）
```python
# 时间复杂度 O(n)
seen = {}  # 维护左侧元素
for j, x in enumerate(nums):
    # 在左侧查找
    if target - x in seen:
        return [seen[target - x], j]
    # 更新哈希表
    seen[x] = j
```

## 🎯 适用场景

- 两数之和、两数之差、两数之积
- 需要在数组中找两个满足某种关系的元素
- 可以通过一次遍历解决的问题

## 📐 算法模板

### 模板1: 两数之和类型

```python
def two_sum_type(nums: List[int], target: int) -> ...:
    """
    枚举右，维护左的基本模板
    """
    seen = {}  # 或 set()，根据需要选择
    
    for j, x in enumerate(nums):
        # 查找左侧是否有满足条件的元素
        complement = target - x  # 根据题目调整
        if complement in seen:
            # 找到答案，进行处理
            ...
        
        # 维护哈希表
        seen[x] = j  # 或 seen.add(x)
    
    return ...
```

### 模板2: 统计数对类型

```python
def count_pairs(nums: List[int]) -> int:
    """
    统计满足某种条件的数对个数
    """
    count = 0
    freq = {}  # 频率哈希表
    
    for x in nums:
        # 查找左侧满足条件的元素个数
        target = ...  # 根据题目计算目标值
        if target in freq:
            count += freq[target]
        
        # 更新频率
        freq[x] = freq.get(x, 0) + 1
    
    return count
```

### 模板3: 维护最值类型

```python
def max_value_with_constraint(nums: List[int]) -> int:
    """
    在满足约束条件下求最值
    """
    max_val = float('-inf')
    left_data = ...  # 维护左侧的某种数据结构
    
    for j, x in enumerate(nums):
        # 利用左侧信息计算当前答案
        if left_data:
            current = ...  # 根据题目计算
            max_val = max(max_val, current)
        
        # 更新左侧信息
        ...
    
    return max_val
```

## 🎓 关键技巧

### 1. 选择合适的数据结构

| 需求 | 数据结构 | 说明 |
|------|---------|------|
| 查找元素是否存在 | `set()` | 只需判断存在性 |
| 记录元素及其位置 | `dict` | 需要下标信息 |
| 统计元素频率 | `Counter` 或 `dict` | 需要计数 |
| 维护最值 | `变量` 或 `有序结构` | 根据具体需求 |

### 2. 一次遍历的要点

```python
# 关键：先使用，再更新
for j, x in enumerate(nums):
    # 1. 先利用左侧信息（查找/计算）
    if target - x in seen:
        ...
    
    # 2. 再更新左侧信息
    seen[x] = j
```

**为什么？** 避免使用当前元素与自己配对。

### 3. 特殊情况处理

#### 处理重复元素
```python
# 如果需要避免重复
if x in seen:
    continue
seen.add(x)
```

#### 处理下标约束
```python
# 例如：j - i <= k
if x in seen and j - seen[x] <= k:
    ...
```

## 📊 复杂度分析

- **时间复杂度**: O(n) - 一次遍历
- **空间复杂度**: O(n) - 哈希表存储

## 💭 常见变体

### 变体1: 两数之和 = target
```python
seen = {}
for j, x in enumerate(nums):
    if target - x in seen:
        return [seen[target - x], j]
    seen[x] = j
```

### 变体2: 两数之差 = k
```python
seen = set()
for x in nums:
    if x - k in seen or x + k in seen:
        return True
    seen.add(x)
```

### 变体3: 两数之积 = target
```python
seen = set()
for x in nums:
    if target % x == 0 and target // x in seen:
        return True
    seen.add(x)
```

### 变体4: 最大差值（维护最小值）
```python
min_val = float('inf')
max_diff = 0
for x in nums:
    max_diff = max(max_diff, x - min_val)
    min_val = min(min_val, x)
```

## 🎯 学习建议

1. **从经典开始**: 先做"两数之和"，理解基本思路
2. **一次遍历**: 尽量用一次遍历实现
3. **注意细节**: 
   - 先使用后更新
   - 避免自己和自己配对
   - 处理重复元素
4. **掌握变体**: 两数之和/差/积/商的不同处理方式

## 🔗 相关资源

- [§0.1.1 基础题目](basics/README.md)
- [§0.1.2 进阶题目](advanced/README.md)
- 回到 [枚举专题主页](../README.md)
