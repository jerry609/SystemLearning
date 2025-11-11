# §2.5 最大化最小值 - 题目列表

> 提升下界，看看能否满足条件

## 题目列表

| 题号 | 题目 | 难度 | 难度分 | 说明 |
|------|------|------|--------|------|
| 3281 | [范围内整数的最大得分](https://leetcode.cn/problems/maximize-score-of-numbers-in-ranges/) | 🟠 中等 | 1768 | 最大化最小间隔 |
| 3620 | [恢复网络路径](https://leetcode.cn/problems/restore-network-paths/) | 🔴 困难 | 1998 | 最大化最小带宽 |
| 2517 | [礼盒的最大甜蜜度](https://leetcode.cn/problems/maximum-tastiness-of-candy-basket/) | 🟠 中等 | 2021 | 最大化最小差值 |
| 1552 | [两球之间的磁力](https://leetcode.cn/problems/magnetic-force-between-two-balls/) | 🟠 中等 | - | **同2517题** |
| 3710 | [最大划分因子](https://leetcode.cn/problems/maximum-partition-factor/) | 🔴 困难 | 2135 | 最大化最小划分 |
| 2812 | [找出最安全路径](https://leetcode.cn/problems/find-the-safest-path-in-a-grid/) | 🟠 中等 | 2154 | 最大化路径最小安全系数 |
| 2528 | [最大化城市的最小电量](https://leetcode.cn/problems/maximize-the-minimum-powered-city/) | 🔴 困难 | 2236 | 最大化最小电量 |
| 3600 | [升级后最大生成树稳定性](https://leetcode.cn/problems/maximum-spanning-tree-stability-after-upgrades/) | 🔴 困难 | 2301 | **做法不止一种** |
| 3449 | [最大化游戏分数的最小值](https://leetcode.cn/problems/maximize-minimum-value-in-game-score/) | 🔴 困难 | 2748 | 最大化最小分数 |
| 3464 | [正方形上的点之间的最大距离](https://leetcode.cn/problems/maximum-distance-between-points-on-a-square/) | 🔴 困难 | 2806 | 最大化最小距离 |
| 1102 | [得分最高的路径](https://leetcode.cn/problems/path-with-maximum-minimum-value/)🔒 | 🟠 中等 | - | **会员题** |
| 1231 | [分享巧克力](https://leetcode.cn/problems/divide-chocolate/)🔒 | 🔴 困难 | - | **会员题** |

## 重点题目详解

### ⭐ 2517. 礼盒的最大甜蜜度

**题意**: 从 price 中选 k 个数，最大化相邻数之间的最小差值。

**核心思路**:
```python
def check(min_diff):
    # 判断能否选出 k 个数，使得相邻差值 >= min_diff
    # 贪心：排序后尽量选
    price.sort()
    count = 1  # 至少选第一个
    last = price[0]
    
    for i in range(1, len(price)):
        if price[i] - last >= min_diff:
            count += 1
            last = price[i]
    
    return count >= k

# 二分最小差值：[0, (max-min) // (k-1)]
# 注意：求的是最大，所以 check 为 true 时更新 left
```

**关键点**:
- 先排序
- 贪心地选择：满足间隔就选
- 间隔越小越容易选够 k 个（单调性）

### ⭐ 1552. 两球之间的磁力

**题意**: 在 position 中选 m 个位置放球，最大化最近两球的距离。

**核心思路**: **完全同 2517 题**

```python
def check(min_dist):
    position.sort()
    count = 1
    last = position[0]
    
    for i in range(1, len(position)):
        if position[i] - last >= min_dist:
            count += 1
            last = position[i]
    
    return count >= m
```

### ⭐ 2812. 找出最安全路径

**题意**: 网格中有小偷，求从左上到右下的路径，最大化路径上点到最近小偷的最小距离。

**核心思路**:
```python
def check(min_safe_dist):
    # 预处理：每个格子到最近小偷的距离（BFS）
    # 然后 BFS 只走距离 >= min_safe_dist 的格子
    visited = set()
    queue = [(0, 0)]
    
    if dist[0][0] < min_safe_dist:
        return False
    
    visited.add((0, 0))
    
    while queue:
        x, y = queue.pop(0)
        if x == m-1 and y == n-1:
            return True
        
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
            nx, ny = x+dx, y+dy
            if (0 <= nx < m and 0 <= ny < n and 
                (nx, ny) not in visited and 
                dist[nx][ny] >= min_safe_dist):
                visited.add((nx, ny))
                queue.append((nx, ny))
    
    return False

# 二分最小安全距离
```

**关键点**:
1. 先用多源 BFS 预处理每个格子到小偷的最短距离
2. 二分最小安全距离
3. check 时只走安全距离 >= 下界的格子

### ⭐ 2528. 最大化城市的最小电量

**题意**: 每个城市有发电站，可以新建 k 个发电站，最大化所有城市电量的最小值。

**核心思路**:
```python
def check(min_power):
    # 贪心：从左到右，电量不足就尽量往右放发电站
    # 差分数组优化
    needed = 0
    diff = [0] * (n + 1)
    current = initial_power[0]  # 初始电量前缀和
    
    for i in range(n):
        current += diff[i]  # 应用差分
        
        if current < min_power:
            # 需要补充 min_power - current
            need = min_power - current
            needed += need
            
            if needed > k:
                return False
            
            # 在 i + r 位置放发电站（影响 [i, min(n-1, i+2r)]）
            current += need
            diff[min(n, i + 2*r + 1)] -= need
    
    return True

# 二分最小电量
```

**关键点**:
- 贪心：电量不足时，尽量往右放（影响更多后续城市）
- 差分数组优化区间更新

## 题目分类

### 1. 间隔/距离问题（4题）

选择若干元素，最大化最小间隔：
- 2517, 1552, 3281, 3464

**核心模式**:
```python
# 排序 + 贪心选择
def check(min_gap):
    count = 1
    last = arr[0]
    for i in range(1, len(arr)):
        if arr[i] - last >= min_gap:
            count += 1
            last = arr[i]
    return count >= k
```

### 2. 路径问题（2题）

最大化路径上的最小值：
- 2812, 1102

**核心**: BFS/DFS 只走满足条件的边/点

### 3. 资源分配（2题）

分配资源，最大化最小分配：
- 2528, 1231

**核心**: 贪心分配 + 差分优化

### 4. 图/树问题（3题）

图或树上的优化问题：
- 3620, 3710, 3600

## Check 函数模式

### 模式1: 贪心选择（最常见）

```python
def check(lower):
    # 排序后贪心选择
    arr.sort()
    count = 1
    last = arr[0]
    
    for x in arr[1:]:
        if x - last >= lower:
            count += 1
            last = x
    
    return count >= target
```

### 模式2: BFS/DFS 路径验证

```python
def check(lower):
    # 只走值 >= lower 的格子/边
    # BFS/DFS 判断能否到达终点
    return has_path_with_limit(lower)
```

### 模式3: 贪心分配

```python
def check(lower):
    # 贪心地分配资源，使每个单位 >= lower
    resources = k
    for item in items:
        if item < lower:
            need = lower - item
            resources -= need
            if resources < 0:
                return False
    return True
```

## 常见技巧

### 技巧1: 排序是关键

大多数"最大化最小间隔"问题需要排序：
```python
arr.sort()
# 然后贪心选择
```

### 技巧2: 差分数组优化

区间更新使用差分数组：
```python
diff = [0] * (n + 1)
# 区间 [l, r] 加 val
diff[l] += val
diff[r + 1] -= val

# 还原
for i in range(1, n):
    diff[i] += diff[i-1]
```

### 技巧3: 多源 BFS 预处理

```python
# 预处理每个点到多个源点的最短距离
def multi_source_bfs(sources):
    queue = list(sources)
    dist = [[INF] * n for _ in range(m)]
    
    for x, y in sources:
        dist[x][y] = 0
    
    while queue:
        x, y = queue.pop(0)
        for dx, dy in directions:
            nx, ny = x+dx, y+dy
            if valid(nx, ny) and dist[nx][ny] == INF:
                dist[nx][ny] = dist[x][y] + 1
                queue.append((nx, ny))
    
    return dist
```

## 练习建议

1. **对比最小化最大值**：理解两者的差异
2. **记住求最大写法**：check 为 true 更新 left，返回 left
3. **掌握贪心策略**：尤其是间隔类问题的贪心
4. **注意预处理**：路径问题通常需要预处理距离

## 常见错误

1. ❌ 用求最小的写法（更新 right 而不是 left）
2. ❌ 返回 right 而不是 left
3. ❌ 贪心策略错误（比如不先排序）
4. ❌ check 函数逻辑反了

## 记忆要点

⚠️ **最重要的区别**：

| 目标 | check 为 true 时 | 返回值 |
|------|------------------|--------|
| 最小化最大值 | `right = mid` | `right` |
| 最大化最小值 | `left = mid` | `left` |

**助记**: 最大化最小值 = 求最大 = 求最大的二分写法

---

**返回**: [最大化最小值](README.md) | [二分答案](../README.md)
