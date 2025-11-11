# 三、贡献法题目列表

## 📝 核心题目

### 1. [907. 子数组的最小值之和](https://leetcode.cn/problems/sum-of-subarray-minimums/)
**难度**: 中等  
**题目难度**: 1976  
**标签**: 贡献法入门  
**描述**: 计算所有子数组的最小值之和  
**重要性**: ⭐⭐⭐⭐⭐ 必做题，贡献法最经典的题目  
**关键点**:
- 使用单调栈找到每个元素作为最小值的范围
- 计算该元素对多少个子数组有贡献
- 注意避免重复计数（相等元素）

---

### 2. [2104. 子数组范围和](https://leetcode.cn/problems/sum-of-subarray-ranges/)
**难度**: 中等  
**题目难度**: 约 2000  
**标签**: 贡献法 + 最大最小值  
**描述**: 计算所有子数组的（最大值 - 最小值）之和  
**提示**: O(n) 做法难度大约 2000  
**关键点**:
- 分别计算最大值之和与最小值之和
- 结果 = 最大值之和 - 最小值之和
- 需要使用两次单调栈（递增和递减）

---

### 3. [1856. 子数组最小乘积的最大值](https://leetcode.cn/problems/maximum-subarray-min-product/)
**难度**: 中等  
**题目难度**: 2051  
**标签**: 贡献法 + 前缀和  
**描述**: 找到子数组的（最小值 × 和）的最大值  
**重要性**: ⭐⭐⭐⭐ 贡献法结合前缀和的典型  
**关键点**:
- 使用单调栈找到每个元素作为最小值的范围
- 使用前缀和快速计算子数组和
- 枚举每个元素作为最小值，计算最大乘积

---

### 4. [2818. 操作使得分最大](https://leetcode.cn/problems/apply-operations-to-maximize-score/)
**难度**: 困难  
**题目难度**: 2397  
**标签**: 贡献法 + 排序 + 贪心  
**描述**: 通过操作使得分最大化  
**关键点**:
- 结合贡献法和排序
- 贪心选择操作顺序

---

### 5. [2281. 巫师的总力量和](https://leetcode.cn/problems/sum-of-total-strength-of-wizards/)
**难度**: 困难  
**题目难度**: 2621  
**标签**: 贡献法 + 前缀和的前缀和  
**描述**: 计算所有子数组的（最小值 × 和）之和  
**重要性**: ⭐⭐⭐⭐⭐ 贡献法最难题目，需要深入理解  
**关键点**:
- 需要快速计算所有以某个元素为最小值的子数组的和的和
- 使用前缀和的前缀和（二阶前缀和）
- 推导复杂的数学公式

**公式推导**:
```
对于 nums[i] 作为最小值的所有子数组:
- 左边界范围: [left[i]+1, i]
- 右边界范围: [i, right[i]-1]
- 需要计算: ∑∑(sum[l..r]) 对所有 left[i]<l≤i≤r<right[i]

使用前缀和: prefix[i] = sum(nums[0..i])
使用前缀和的前缀和: prePrefix[i] = sum(prefix[0..i])
```

---

### 6. [3430. 最多 K 个元素的子数组的最值之和](https://leetcode.cn/problems/find-the-sum-of-the-power-of-all-subsequences/)
**难度**: 困难  
**题目难度**: 2645  
**标签**: 贡献法 + 约束条件  
**描述**: 在限制条件下计算最值之和

---

### 7. [3359. 查找最大元素不超过 K 的有序子矩阵](https://leetcode.cn/problems/find-the-maximum-number-of-elements-in-subset/)
**难度**: 困难  
**标签**: 贡献法 + 矩形  
**会员题**  
**描述**: 结合矩形和贡献法的问题

---

## 🌟 思维扩展题目

### 8. [2334. 元素值大于变化阈值的子数组](https://leetcode.cn/problems/subarray-with-elements-greater-than-varying-threshold/)
**难度**: 困难  
**题目难度**: 2381  
**标签**: 单调栈 + 并查集  
**描述**: 找到满足特定阈值条件的子数组  
**关键点**:
- 使用单调栈确定范围
- 结合并查集优化查找

---

### 9. [2962. 统计最大元素出现至少 K 次的子数组](https://leetcode.cn/problems/count-subarrays-where-max-element-appears-at-least-k-times/)
**难度**: 中等  
**标签**: 滑动窗口 + 思考题  
**描述**: 统计最大元素至少出现 K 次的子数组  
**附加**: 包含思考题（解答见评论区）

---

## 📊 学习进度

### 必做核心题目
- [ ] 907. 子数组的最小值之和 ⭐⭐⭐⭐⭐
- [ ] 1856. 子数组最小乘积的最大值 ⭐⭐⭐⭐
- [ ] 2281. 巫师的总力量和 ⭐⭐⭐⭐⭐

### 重要题目
- [ ] 2104. 子数组范围和
- [ ] 2818. 操作使得分最大

### 高级题目
- [ ] 3430. 最多 K 个元素的子数组的最值之和
- [ ] 3359. 查找最大元素不超过 K 的有序子矩阵（会员题）

### 思维扩展
- [ ] 2334. 元素值大于变化阈值的子数组
- [ ] 2962. 统计最大元素出现至少 K 次的子数组

## 💡 学习路线

### 阶段一：理解贡献法（必做）

#### 1. [907. 子数组的最小值之和](https://leetcode.cn/problems/sum-of-subarray-minimums/) ⭐⭐⭐⭐⭐
**为什么先做这题？**
- 最纯粹的贡献法问题
- 理解核心思想：枚举元素而非枚举子数组
- 掌握如何避免重复计数

**学习要点**:
```python
# 核心思路
for i in range(n):
    # nums[i] 作为最小值的子数组数量
    left_count = i - left[i]      # 左边界选择数
    right_count = right[i] - i    # 右边界选择数
    contribution = nums[i] * left_count * right_count
```

**常见错误**:
- 相等元素重复计数
- 边界处理不当

---

### 阶段二：结合前缀和

#### 2. [1856. 子数组最小乘积的最大值](https://leetcode.cn/problems/maximum-subarray-min-product/) ⭐⭐⭐⭐
**为什么是第二题？**
- 在 907 的基础上增加了"子数组和"
- 学习如何结合前缀和快速计算

**学习要点**:
```python
# 预处理前缀和
prefix = [0]
for x in nums:
    prefix.append(prefix[-1] + x)

# 对于 nums[i] 作为最小值
for i in range(n):
    # 子数组范围 [left[i]+1, right[i]-1]
    subarray_sum = prefix[right[i]] - prefix[left[i]+1]
    min_product = nums[i] * subarray_sum
```

---

#### 3. [2104. 子数组范围和](https://leetcode.cn/problems/sum-of-subarray-ranges/)
**学习要点**:
- 分别计算最大值贡献和最小值贡献
- 使用两个单调栈（递增和递减）
- 理解"范围"可以拆分成两个独立问题

---

### 阶段三：挑战高难度

#### 4. [2281. 巫师的总力量和](https://leetcode.cn/problems/sum-of-total-strength-of-wizards/) ⭐⭐⭐⭐⭐
**为什么最难？**
- 需要计算"所有子数组和的和"
- 引入前缀和的前缀和（二阶前缀和）
- 推导复杂的数学公式

**学习要点**:
```python
# 前缀和
prefix[i] = sum(nums[0..i])

# 前缀和的前缀和
pre_prefix[i] = sum(prefix[0..i])

# 对于 nums[i] 作为最小值，需要计算:
# ∑∑(sum[l..r]) for all left[i] < l <= i <= r < right[i]

# 利用前缀和的性质:
# sum[l..r] = prefix[r] - prefix[l-1]

# 进一步化简，使用 pre_prefix 加速计算
```

**建议**:
- 先在纸上推导公式
- 理解为什么需要二阶前缀和
- 可以先看题解理解思路，再自己实现

---

### 阶段四：扩展应用

#### 5. [2818. 操作使得分最大](https://leetcode.cn/problems/apply-operations-to-maximize-score/)
- 贡献法 + 排序 + 贪心

#### 6. [2334. 元素值大于变化阈值的子数组](https://leetcode.cn/problems/subarray-with-elements-greater-than-varying-threshold/)
- 贡献法 + 并查集

---

## 🎯 核心技巧总结

### 1. 避免重复计数
```python
# 左边用严格小于，右边用小于等于（或相反）
# 左边
while st and arr[st[-1]] > arr[i]:   # 严格大于
    st.pop()

# 右边  
while st and arr[st[-1]] >= arr[i]:  # 大于等于
    st.pop()
```

### 2. 前缀和加速
```python
# 一阶前缀和：快速计算子数组和
prefix[i] = sum(nums[0..i])
sum(nums[l..r]) = prefix[r] - prefix[l-1]

# 二阶前缀和：快速计算多个子数组和的和
pre_prefix[i] = sum(prefix[0..i])
```

### 3. 贡献计算公式
```python
# 基础贡献（907）
contribution = nums[i] * (i - left[i]) * (right[i] - i)

# 结合子数组和（1856）
subarray_sum = prefix[right[i]] - prefix[left[i]+1]
contribution = nums[i] * subarray_sum

# 所有子数组和（2281）
# 需要推导复杂公式，使用二阶前缀和
```

## 🔍 调试技巧

1. **小数据测试**: 用长度为 3-5 的数组手动验证
2. **边界检查**: 特别注意 `left[i] = -1` 和 `right[i] = n` 的情况
3. **模数运算**: 大数问题记得取模 `10^9 + 7`
4. **输出中间结果**: 打印 `left`, `right` 数组检查正确性

## 🔗 相关资源

- 回到 [贡献法专题](README.md)
- 回到 [单调栈主页](../README.md)
- 继续学习 [最小字典序](../lexicographic/README.md)

## 📚 推荐阅读顺序

1. ⭐ 907 - 理解贡献法
2. ⭐ 1856 - 学习前缀和结合
3. 2104 - 练习最大最小值分离
4. ⭐ 2281 - 挑战最难题目
5. 其他题目 - 按兴趣选择
