# 四、其他 - 题目列表

## 题目列表

| 题号 | 题目 | 难度 | 说明 |
|------|------|------|------|
| 69 | [x 的平方根](https://leetcode.cn/problems/sqrtx/) | 🟢 简单 | 二分求最大的 m，满足 m² ≤ x |
| 74 | [搜索二维矩阵](https://leetcode.cn/problems/search-a-2d-matrix/) | 🟠 中等 | 把二维当作一维二分 |
| 278 | [第一个错误的版本](https://leetcode.cn/problems/first-bad-version/) | 🟢 简单 | lowerBound 应用 |
| 374 | [猜数字大小](https://leetcode.cn/problems/guess-number-higher-or-lower/) | 🟢 简单 | 标准二分 |
| 162 | [寻找峰值](https://leetcode.cn/problems/find-peak-element/) | 🟠 中等 | 峰值查找 |
| 1901 | [寻找峰值 II](https://leetcode.cn/problems/find-a-peak-element-ii/) | 🟠 中等 | 二维峰值 |
| 852 | [山脉数组的峰顶索引](https://leetcode.cn/problems/peak-index-in-a-mountain-array/) | 🟠 中等 | 山脉峰值 |
| 1095 | [山脉数组中查找目标值](https://leetcode.cn/problems/find-in-mountain-array/) | 🔴 困难 | 1827 - 先找峰，再两侧二分 |
| 153 | [寻找旋转排序数组中的最小值](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/) | 🟠 中等 | 旋转数组最小值 |
| 154 | [寻找旋转排序数组中的最小值 II](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array-ii/) | 🔴 困难 | 有重复元素 |
| 33 | [搜索旋转排序数组](https://leetcode.cn/problems/search-in-rotated-sorted-array/) | 🟠 中等 | 旋转数组搜索 |
| 81 | [搜索旋转排序数组 II](https://leetcode.cn/problems/search-in-rotated-sorted-array-ii/) | 🟠 中等 | 有重复元素 |
| 222 | [完全二叉树的节点个数](https://leetcode.cn/problems/count-complete-tree-nodes/) | 🟢 简单 | 二分+树 |
| 1539 | [第 k 个缺失的正整数](https://leetcode.cn/problems/kth-missing-positive-number/) | 🟢 简单 | 二分缺失数 |
| 540 | [有序数组中的单一元素](https://leetcode.cn/problems/single-element-in-a-sorted-array/) | 🟠 中等 | 异或性质+二分 |
| 4 | [寻找两个正序数组的中位数](https://leetcode.cn/problems/median-of-two-sorted-arrays/) | 🔴 困难 | 二分+中位数 |
| 1064 | [不动点](https://leetcode.cn/problems/fixed-point/)🔒 | 🟢 简单 | **会员题** |
| 702 | [搜索长度未知的有序数组](https://leetcode.cn/problems/search-in-a-sorted-array-of-unknown-size/)🔒 | 🟠 中等 | **会员题** |
| 2936 | [包含相等值数字块的数量](https://leetcode.cn/problems/number-of-equal-numbers-blocks/)🔒 | 🟠 中等 | **会员题** |
| 1060 | [有序数组中的缺失元素](https://leetcode.cn/problems/missing-element-in-sorted-array/)🔒 | 🟠 中等 | **会员题** |
| 1198 | [找出所有行中最小公共元素](https://leetcode.cn/problems/find-smallest-common-element-in-all-rows/)🔒 | 🟠 中等 | **会员题** |
| 1428 | [至少有一个 1 的最左端列](https://leetcode.cn/problems/leftmost-column-with-at-least-a-one/)🔒 | 🟠 中等 | **会员题** |
| 1533 | [找到最大整数的索引](https://leetcode.cn/problems/find-the-index-of-the-large-integer/)🔒 | 🟠 中等 | **会员题** |
| 2387 | [行排序矩阵的中位数](https://leetcode.cn/problems/median-of-a-row-wise-sorted-matrix/)🔒 | 🟠 中等 | **会员题** |
| 302 | [包含全部黑色像素的最小矩形](https://leetcode.cn/problems/smallest-rectangle-enclosing-black-pixels/)🔒 | 🔴 困难 | **会员题** |

## 题目分类

### 1. 基础变体（4题）

| 题号 | 题目 | 核心思路 |
|------|------|----------|
| 69 | x 的平方根 | 二分求最大的 m，满足 m² ≤ x |
| 74 | 搜索二维矩阵 | 把 (i, j) 映射到 i*n+j |
| 278 | 第一个错误的版本 | lowerBound |
| 374 | 猜数字大小 | 标准二分 |

### 2. 峰值查找（4题）

**核心**: 利用梯度方向缩小范围

| 题号 | 题目 | 核心思路 |
|------|------|----------|
| 162 | 寻找峰值 | 向更大的邻居方向移动 |
| 1901 | 寻找峰值 II | 二维峰值，先找行最大，再二分列 |
| 852 | 山脉数组的峰顶索引 | 单峰数组 |
| 1095 | 山脉数组中查找目标值 | 先找峰，再分两段二分 |

**162. 寻找峰值**:
```python
def findPeakElement(nums):
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[mid + 1]:
            right = mid  # 峰在左边（包括mid）
        else:
            left = mid + 1  # 峰在右边
    return left
```

### 3. 旋转数组（4题）

**核心**: 判断哪一段是有序的

| 题号 | 题目 | 核心思路 |
|------|------|----------|
| 153 | 寻找旋转排序数组中的最小值 | 与 right 比较 |
| 154 | 寻找旋转排序数组中的最小值 II | 有重复，right-- |
| 33 | 搜索旋转排序数组 | 判断哪段有序 |
| 81 | 搜索旋转排序数组 II | 有重复，特殊处理 |

**153. 寻找旋转排序数组中的最小值**:
```python
def findMin(nums):
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[right]:
            left = mid + 1  # 最小值在右边
        else:
            right = mid  # 最小值在左边（包括mid）
    return nums[left]
```

**33. 搜索旋转排序数组**:
```python
def search(nums, target):
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if nums[mid] == target:
            return mid
        
        # 判断哪一段是有序的
        if nums[left] <= nums[mid]:  # 左段有序
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:  # 右段有序
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1
```

### 4. 特殊结构（3题）

| 题号 | 题目 | 核心思路 |
|------|------|----------|
| 222 | 完全二叉树的节点个数 | 二分层数 |
| 1539 | 第 k 个缺失的正整数 | 二分位置 |
| 540 | 有序数组中的单一元素 | 利用下标奇偶性 |

**222. 完全二叉树的节点个数**:
```python
def countNodes(root):
    if not root:
        return 0
    
    # 计算树的深度
    def get_depth(node):
        depth = 0
        while node.left:
            depth += 1
            node = node.left
        return depth
    
    left_depth = get_depth(root.left)
    right_depth = get_depth(root.right)
    
    if left_depth == right_depth:
        # 左子树是满二叉树
        return (1 << left_depth) + countNodes(root.right)
    else:
        # 右子树是满二叉树
        return (1 << right_depth) + countNodes(root.left)
```

**540. 有序数组中的单一元素**:
```python
def singleNonDuplicate(nums):
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = (left + right) // 2
        
        # 确保 mid 是偶数
        if mid % 2 == 1:
            mid -= 1
        
        # 检查 mid 和 mid+1
        if nums[mid] == nums[mid + 1]:
            left = mid + 2  # 单一元素在右边
        else:
            right = mid  # 单一元素在左边（包括mid）
    
    return nums[left]
```

### 5. 高级应用（2题）

| 题号 | 题目 | 核心思路 |
|------|------|----------|
| 4 | 寻找两个正序数组的中位数 | 二分第一个数组的分割点 |
| 1064 | 不动点 | arr[i] == i |

**4. 寻找两个正序数组的中位数**:
```python
def findMedianSortedArrays(nums1, nums2):
    # 确保 nums1 是较短的数组
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    
    m, n = len(nums1), len(nums2)
    left, right = 0, m
    
    while left <= right:
        i = (left + right) // 2
        j = (m + n + 1) // 2 - i
        
        # 边界处理
        maxLeft1 = float('-inf') if i == 0 else nums1[i - 1]
        minRight1 = float('inf') if i == m else nums1[i]
        maxLeft2 = float('-inf') if j == 0 else nums2[j - 1]
        minRight2 = float('inf') if j == n else nums2[j]
        
        if maxLeft1 <= minRight2 and maxLeft2 <= minRight1:
            # 找到正确的分割
            if (m + n) % 2 == 0:
                return (max(maxLeft1, maxLeft2) + min(minRight1, minRight2)) / 2
            else:
                return max(maxLeft1, maxLeft2)
        elif maxLeft1 > minRight2:
            right = i - 1
        else:
            left = i + 1
```

## 重点题目

### ⭐ 162. 寻找峰值

**关键**: 局部最优 → 向更大的邻居方向移动

### ⭐ 153. 寻找旋转排序数组中的最小值

**关键**: 与 right 比较，判断最小值在哪边

### ⭐ 33. 搜索旋转排序数组

**关键**: 先判断哪一段是有序的

### ⭐ 540. 有序数组中的单一元素

**关键**: 利用下标奇偶性和对称性

### ⭐ 4. 寻找两个正序数组的中位数

**关键**: 二分第一个数组的分割点

## 练习建议

1. **分类练习**：按类型逐个攻克
2. **理解性质**：每道题的特殊性质是什么
3. **注意边界**：这些题目边界条件较多
4. **对比标准二分**：理解如何变化

## 常见技巧

### 技巧1: 二维转一维

```python
# 二维坐标 (i, j) → 一维
index = i * n + j

# 一维 index → 二维
i = index // n
j = index % n
```

### 技巧2: 判断旋转数组的有序段

```python
if nums[left] <= nums[mid]:
    # 左段有序
else:
    # 右段有序
```

### 技巧3: 峰值查找的方向选择

```python
if nums[mid] > nums[mid + 1]:
    # 峰在左边（包括 mid）
    right = mid
else:
    # 峰在右边
    left = mid + 1
```

---

**返回**: [其他](README.md) | [二分查找专题](../README.md)
