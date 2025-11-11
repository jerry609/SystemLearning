# §0.2 枚举中间

## 📖 核心思想

对于三个或四个变量的问题，**枚举中间的变量往往更好算**。

### 为什么？

比如问题有三个下标，需要满足 $0 \leq i < j < k < n$

**对比两种方案**:

#### 方案1: 枚举 i（枚举左）
```python
for i in range(n):
    # 后续计算中还需保证 j < k
    for j in range(i + 1, n):
        for k in range(j + 1, n):
            ...
```
- ❌ 需要三层循环
- ❌ 还要维护 j < k 的关系

#### 方案2: 枚举 j（枚举中间）
```python
for j in range(n):
    # i 和 k 自动被 j 隔开
    # 左侧：i < j
    # 右侧：k > j
    # 无需关心 i 和 k 的位置关系
    ...
```
- ✅ i 和 k 自动被 j 隔开，互相独立
- ✅ 可以分别处理左右两侧
- ✅ 后续计算中无需关心 i 和 k 的位置关系

## 💡 适用场景

### 场景1: 三元组问题
满足 $i < j < k$ 的三元组问题：
- 山形三元组
- 回文子序列
- 直角三角形
- 等等

### 场景2: 路径问题
需要经过中间节点的路径：
- 树的路径
- 图的路径
- 等等

### 场景3: 区间问题
需要分割点的问题：
- 左右区间的某种性质
- 中心扩展
- 等等

## 📐 算法模板

### 模板1: 基础三元组

```python
def solve_triplet(nums: List[int]) -> ...:
    """
    枚举中间元素 j，分别维护左右信息
    """
    n = len(nums)
    result = 0
    
    for j in range(1, n - 1):  # 枚举中间
        # 处理左侧 (i < j)
        left_info = ...
        for i in range(j):
            left_info = update_left(left_info, nums[i])
        
        # 处理右侧 (k > j)
        right_info = ...
        for k in range(j + 1, n):
            right_info = update_right(right_info, nums[k])
        
        # 利用左右信息计算答案
        result += calculate(left_info, nums[j], right_info)
    
    return result
```

### 模板2: 预处理优化

```python
def solve_triplet_optimized(nums: List[int]) -> ...:
    """
    预处理左右信息，避免重复计算
    """
    n = len(nums)
    
    # 预处理左侧信息
    left = [None] * n
    left[0] = initial_left()
    for i in range(1, n):
        left[i] = update_left(left[i-1], nums[i-1])
    
    # 预处理右侧信息
    right = [None] * n
    right[n-1] = initial_right()
    for i in range(n - 2, -1, -1):
        right[i] = update_right(right[i+1], nums[i+1])
    
    # 枚举中间
    result = 0
    for j in range(1, n - 1):
        result += calculate(left[j], nums[j], right[j])
    
    return result
```

### 模板3: 回旋镖模式（中心枚举）

```python
def solve_with_center(points: List[...]) -> ...:
    """
    以某个点为中心，统计满足条件的组合
    """
    result = 0
    
    for center in points:  # 枚举中心
        groups = {}  # 按某种特征分组
        
        for point in points:
            if point == center:
                continue
            
            # 计算特征（如距离、角度等）
            feature = calculate_feature(center, point)
            
            # 利用已有信息计算答案
            result += process(groups, feature)
            
            # 更新分组信息
            groups[feature] = groups.get(feature, 0) + 1
    
    return result
```

## 🎯 关键技巧

### 技巧1: 左右分离
```python
# 左侧维护最小值/最大值/计数等
left_max = [0] * n
for i in range(1, n):
    left_max[i] = max(left_max[i-1], nums[i-1])

# 右侧维护最小值/最大值/计数等  
right_min = [0] * n
for i in range(n - 2, -1, -1):
    right_min[i] = min(right_min[i+1], nums[i+1])

# 枚举中间
for j in range(1, n - 1):
    if left_max[j] < nums[j] < right_min[j]:
        count += 1
```

### 技巧2: 中心扩展
```python
# 以每个元素为中心
for center in range(n):
    # 统计左右两侧满足条件的元素
    left_count = count_left(nums, center)
    right_count = count_right(nums, center)
    result += left_count * right_count  # 组合数
```

### 技巧3: 按特征分组
```python
# 以某个点为中心，按特征分组
for center in range(n):
    groups = {}
    for other in range(n):
        if other == center:
            continue
        feature = get_feature(center, other)
        groups[feature] = groups.get(feature, 0) + 1
    
    # 计算组合数
    for count in groups.values():
        result += count * (count - 1) // 2  # C(count, 2)
```

## 📊 复杂度分析

### 基础三元组
- **时间复杂度**: O(n^2) 或 O(n)（预处理优化）
- **空间复杂度**: O(n)（预处理数组）

### 回旋镖模式
- **时间复杂度**: O(n^2)
- **空间复杂度**: O(n)（哈希表）

## 💭 对比总结

| 枚举方式 | 优点 | 缺点 | 适用场景 |
|---------|------|------|---------|
| 枚举左 (i) | 直观 | 需要维护 j < k | 依赖顺序的问题 |
| 枚举中 (j) | i, k 独立 | 需要预处理 | 多数三元组问题 |
| 枚举右 (k) | 前缀信息 | 反向思考 | 特殊约束问题 |

## 🎓 学习建议

1. **理解为什么**: 先理解为什么枚举中间更简单
2. **掌握预处理**: 学会预处理左右信息避免重复计算
3. **练习分组**: 掌握按特征分组的技巧
4. **对比方法**: 尝试不同枚举方式，体会差异

## 🔗 相关资源

- [题目列表](problems.md)
- 回到 [枚举专题主页](../README.md)
- 参考 [§0.1 枚举右，维护左](../enumerate-right/README.md)
