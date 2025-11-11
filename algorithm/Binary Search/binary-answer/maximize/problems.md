# §2.2 求最大 - 题目列表

> ⚠️ 注意：这些题目的二分写法和"求最小"相反！

## 题目列表

| 题号 | 题目 | 难度 | 难度分 | 说明 |
|------|------|------|--------|------|
| 275 | [H 指数 II](https://leetcode.cn/problems/h-index-ii/) | 🟠 中等 | - | 二分 H 指数 |
| 2226 | [每个小孩最多能分到多少糖果](https://leetcode.cn/problems/maximum-candies-allocated-to-k-children/) | 🟠 中等 | 1646 | 二分糖果数 |
| 2982 | [找出出现至少三次的最长特殊子字符串 II](https://leetcode.cn/problems/find-longest-special-substring-that-occurs-thrice-ii/) | 🟠 中等 | 1773 | 二分长度 |
| 2576 | [求出最多标记下标](https://leetcode.cn/problems/find-the-maximum-number-of-marked-indices/) | 🟠 中等 | 1843 | 二分标记数 |
| 1898 | [可移除字符的最大数目](https://leetcode.cn/problems/maximum-number-of-removable-characters/) | 🟠 中等 | 1913 | 二分移除数 |
| 1802 | [有界数组中指定下标处的最大值](https://leetcode.cn/problems/maximum-value-at-a-given-index-in-a-bounded-array/) | 🟠 中等 | 1929 | 二分最大值 |
| 1642 | [可以到达的最远建筑](https://leetcode.cn/problems/furthest-building-you-can-reach/) | 🟠 中等 | 1962 | 二分建筑数 |
| 2861 | [最大合金数](https://leetcode.cn/problems/maximum-number-of-alloys/) | 🟠 中等 | 1981 | 二分合金数 |
| 3007 | [价值和小于等于 K 的最大数字](https://leetcode.cn/problems/maximum-number-that-sum-of-the-prices-is-less-than-or-equal-to-k/) | 🟠 中等 | 2258 | 二分数字 |
| 2141 | [同时运行 N 台电脑的最长时间](https://leetcode.cn/problems/maximum-running-time-of-n-computers/) | 🔴 困难 | 2265 | 二分时间 |
| 2258 | [逃离火灾](https://leetcode.cn/problems/escape-the-spreading-fire/) | 🔴 困难 | 2347 | 二分等待时间 |
| 2071 | [你可以安排的最多任务数目](https://leetcode.cn/problems/maximum-number-of-tasks-you-can-assign/) | 🔴 困难 | 2648 | 二分任务数 |
| - | [LCP 78. 城墙防线](https://leetcode.cn/problems/城墙防线/) | 🔴 困难 | - | 二分防御值 |
| 1618 | [找出适应屏幕的最大字号](https://leetcode.cn/problems/maximum-font-to-fit-a-sentence-in-a-screen/)🔒 | 🟠 中等 | - | **会员题** |
| 1891 | [割绳子](https://leetcode.cn/problems/cutting-ribbons/)🔒 | 🟠 中等 | - | **会员题** |
| 2137 | [通过倒水操作让所有的水桶所含水量相等](https://leetcode.cn/problems/pour-water-between-buckets-to-make-water-levels-equal/)🔒 | 🔴 困难 | - | **会员题** |
| 3344 | [最大尺寸数组](https://leetcode.cn/problems/maximum-size-subarray/)🔒 | 🔴 困难 | - | **会员题** |
| 644 | [子数组最大平均数 II](https://leetcode.cn/problems/maximum-average-subarray-ii/)🔒 | 🔴 困难 | - | **会员题** |

## 重点题目详解

### ⭐ 275. H 指数 II

**题意**: 数组 citations 递增，求 H 指数（有 h 篇论文至少被引用 h 次）。

**核心思路**:
```python
def check(h):
    # 判断是否有至少 h 篇论文被引用至少 h 次
    # 由于数组递增，只需看最后 h 篇
    return citations[n - h] >= h if h <= n else False

# 二分 H 指数：[0, n]
```

**关键点**:
- H 指数越小越容易满足（单调性）
- 利用数组递增的特性

### ⭐ 2226. 每个小孩最多能分到多少糖果

**题意**: 把糖果堆分给 k 个小孩，每堆可以任意切分，求每个小孩最多能分到多少。

**核心思路**:
```python
def check(x):
    # 判断能否给 k 个小孩每人分 x 个糖果
    total = sum(pile // x for pile in candies)
    return total >= k

# 二分糖果数：[1, max(candies)]
```

**关键点**:
- 每人分的糖果越少越容易满足（单调性）
- 每堆可以分给多个小孩

### ⭐ 1898. 可移除字符的最大数目

**题意**: 从 s 中移除 removable 的前 k 个字符后，p 是否仍是 s 的子序列。

**核心思路**:
```python
def check(k):
    # 标记前 k 个要移除的字符
    removed = set(removable[:k])
    # 判断 p 是否是剩余字符的子序列
    j = 0
    for i, c in enumerate(s):
        if i not in removed and j < len(p) and c == p[j]:
            j += 1
    return j == len(p)

# 二分移除数：[0, len(removable)]
```

**关键点**:
- 移除越少越容易满足（单调性）
- 使用子序列匹配

### ⭐ 2861. 最大合金数

**题意**: 有多种配方制作合金，每种合金需要特定的金属，求最多能制作多少合金。

**核心思路**:
```python
def check(num):
    # 对每种配方，判断能否制作 num 个合金
    for composition in compositions:
        cost = 0
        for i, need in enumerate(composition):
            have = stock[i]
            required = need * num
            if required > have:
                cost += (required - have) * costs[i]
        if cost <= budget:
            return True
    return False

# 二分合金数：[0, 大值]
```

**关键点**:
- 制作越少越容易满足（单调性）
- 只要有一种配方可行即可

### ⭐ 2141. 同时运行 N 台电脑的最长时间

**题意**: 有多块电池，n 台电脑，电池可以在电脑间切换，求最长运行时间。

**核心思路**:
```python
def check(minutes):
    # 判断能否运行 minutes 分钟
    # 需要总电量 >= n * minutes
    # 但每块电池最多贡献 minutes
    total = sum(min(battery, minutes) for battery in batteries)
    return total >= n * minutes

# 二分时间：[0, sum(batteries) // n]
```

**关键点**:
- 时间越短越容易满足（单调性）
- 每块电池的贡献有上限

## 题目分类

### 1. 二分数量（7题）
二分物品数量、任务数等：
- 2226, 2576, 1898, 2861, 2071, 275, 1891

### 2. 二分时间（2题）
二分运行时间、等待时间：
- 2141, 2258

### 3. 二分长度/大小（4题）
二分长度、字号等：
- 2982, 1642, 1618, 644

### 4. 二分值（3题）
二分某个值的大小：
- 1802, 3007, 2137

## Check 函数模式

### 模式1: 贪心分配

```python
def check(x):
    # 贪心地分配资源
    count = 0
    for item in items:
        count += item // x
    return count >= k
```

### 模式2: 计算成本

```python
def check(x):
    # 计算达到 x 需要的成本
    cost = 0
    for item in items:
        cost += calculate_cost(item, x)
    return cost <= budget
```

### 模式3: 验证可行性

```python
def check(x):
    # 验证是否可以达到 x
    # 使用贪心、双指针等算法
    return is_achievable(x)
```

## 常见技巧

### 技巧1: 排序后二分

```python
# 先排序，然后二分
arr.sort()

def check(x):
    # 利用排序的性质
    pass
```

### 技巧2: 多方案取最优

```python
def check(x):
    # 尝试所有方案，只要有一个可行即可
    for plan in plans:
        if try_plan(plan, x):
            return True
    return False
```

### 技巧3: 上限约束

```python
def check(x):
    # 每个资源的贡献有上限
    total = sum(min(resource, x) for resource in resources)
    return total >= target
```

## 练习建议

1. **对比求最小**：理解写法上的差异
2. **记住口诀**：check 更新谁就返回谁
3. **理解单调性**：为什么答案越小越容易满足？
4. **注意循环不变量**：left 和 right 的含义

## 常见错误

1. ❌ 把求最大当成求最小来写
2. ❌ check 为 true 时更新错了变量
3. ❌ 最后返回错了变量
4. ❌ 循环不变量写反

## 记忆技巧

### 开区间写法总结

| 目标 | check 为 true 时 | 返回值 |
|------|------------------|--------|
| 求最小 | `right = mid` | `right` |
| 求最大 | `left = mid` | `left` |

**统一记忆**: check 更新的是谁，最终就返回谁！

---

**返回**: [求最大](README.md) | [二分答案](../README.md)
