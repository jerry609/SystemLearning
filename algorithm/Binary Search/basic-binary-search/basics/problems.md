# §1.1 基础题目

> **推荐**: 先完成34和35题，并阅读题解中的答疑部分

## 题目列表

| 题号 | 题目 | 难度 | 说明 |
|------|------|------|------|
| 34 | [在排序数组中查找元素的第一个和最后一个位置](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/) | 🟠 中等 | **推荐阅读题解中的答疑** |
| 35 | [搜索插入位置](https://leetcode.cn/problems/search-insert-position/) | 🟢 简单 | **推荐阅读题解中的答疑** |
| 704 | [二分查找](https://leetcode.cn/problems/binary-search/) | 🟢 简单 | 最基础的二分查找 |
| 744 | [寻找比目标字母大的最小字母](https://leetcode.cn/problems/find-smallest-letter-greater-than-target/) | 🟢 简单 | lowerBound 的直接应用 |
| 2529 | [正整数和负整数的最大计数](https://leetcode.cn/problems/maximum-count-of-positive-integer-and-negative-integer/) | 🟢 简单 | **做到 O(log n)** |

## 题目详解

### 34. 在排序数组中查找元素的第一个和最后一个位置

**核心思路**:
- 找第一个位置：`lowerBound(nums, target)`
- 找最后一个位置：`lowerBound(nums, target + 1) - 1`

**关键点**:
- 需要判断找到的位置是否有效
- 注意边界情况

### 35. 搜索插入位置

**核心思路**:
- 直接使用 `lowerBound(nums, target)`

**关键点**:
- lowerBound 返回的就是插入位置

### 704. 二分查找

**核心思路**:
- 标准二分查找模板

**关键点**:
- 掌握开区间和闭区间两种写法

### 744. 寻找比目标字母大的最小字母

**核心思路**:
- 找 `> target` 的第一个：`lowerBound(letters, target + 1)`
- 或者找 `>= target + 1` 的第一个

**关键点**:
- 注意字符的循环特性

### 2529. 正整数和负整数的最大计数

**核心思路**:
- 负数个数：`lowerBound(nums, 0)`
- 正数个数：`n - lowerBound(nums, 1)`
- 或：`n - lowerBound(nums, 0)` 减去 0 的个数

**关键点**:
- 时间复杂度必须是 O(log n)

## 练习建议

1. **先做34和35**：这两题是理解 lowerBound 的关键
2. **阅读答疑**：理解为什么这样写
3. **对比写法**：开区间 vs 闭区间
4. **总结规律**：找第一个 vs 找最后一个

## 常见错误

1. ❌ 混淆第一个和最后一个的求法
2. ❌ 忘记检查返回值是否越界
3. ❌ 边界条件 `<` 和 `<=` 混淆
4. ❌ 循环条件 `left < right` 和 `left <= right` 混淆

## 知识点总结

| 需求 | 写法 |
|------|------|
| 第一个 >= x | `lowerBound(nums, x)` |
| 第一个 > x | `lowerBound(nums, x + 1)` |
| 最后一个 < x | `lowerBound(nums, x) - 1` |
| 最后一个 <= x | `lowerBound(nums, x + 1) - 1` |

---

**返回**: [基础](README.md) | [基础二分查找](../README.md)
