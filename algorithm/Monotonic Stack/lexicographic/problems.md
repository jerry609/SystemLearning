# 四、最小字典序题目列表

## 📝 核心题目

### 1. [402. 移掉 K 位数字](https://leetcode.cn/problems/remove-k-digits/)
**难度**: 中等  
**题目难度**: 约 1800  
**标签**: 单调栈 + 贪心  
**描述**: 给定一个数字字符串，移除 k 位数字，使剩余数字最小  
**重要性**: ⭐⭐⭐⭐⭐ 必做题，字典序问题的入门  

**关键点**:
```python
# 贪心策略：尽可能让高位小
while st and k > 0 and st[-1] > digit:
    st.pop()
    k -= 1

# 注意处理：
# 1. 如果 k 还没用完（序列已经递增）
# 2. 去除前导零
# 3. 空字符串返回 "0"
```

**易错点**:
- 忘记去除前导零
- k 没用完的情况
- 结果为空字符串

---

### 2. [1673. 找出最具竞争力的子序列](https://leetcode.cn/problems/find-the-most-competitive-subsequence/)
**难度**: 中等  
**题目难度**: 1802  
**标签**: 单调栈 + 贪心  
**描述**: 从数组中选择 k 个数字，使得子序列字典序最小  
**重要性**: ⭐⭐⭐⭐ 与 402 互补，一个是删除，一个是选择  

**关键点**:
```python
# 关键条件：删除后还能凑够 k 个数
while st and st[-1] > x and len(st) + (n - i) > k:
    st.pop()

# 只在需要时添加
if len(st) < k:
    st.append(x)
```

**对比 402**:
| 题目 | 操作 | 关键条件 |
|------|------|----------|
| 402 | 删除 k 个 | `k > 0` |
| 1673 | 选择 k 个 | `len(st) + 剩余 > k` |

---

### 3. [316. 去除重复字母](https://leetcode.cn/problems/remove-duplicate-letters/)
**难度**: 中等  
**题目难度**: 2185  
**标签**: 单调栈 + 贪心 + 去重  
**描述**: 去除重复字母，使得每个字母只出现一次且字典序最小  
**重要性**: ⭐⭐⭐⭐⭐ 必做题，结合了去重和字典序  

**关键点**:
```python
# 记录每个字符最后出现位置
last = {c: i for i, c in enumerate(s)}

# 使用 set 记录是否在栈中
in_stack = set()

# 核心逻辑
for i, c in enumerate(s):
    if c in in_stack:
        continue  # 已经在栈中，跳过
    
    # 可以删除栈顶的条件：后面还会出现
    while st and st[-1] > c and last[st[-1]] > i:
        in_stack.remove(st.pop())
    
    st.append(c)
    in_stack.add(c)
```

**技巧**:
- 使用 `last` 字典记录最后位置
- 使用 `in_stack` 避免重复添加
- 删除时同步更新 `in_stack`

---

### 4. [316 扩展：重复个数不超过 limit](题目链接待补充)
**难度**: 困难  
**标签**: 单调栈 + 约束条件  
**描述**: 316 题的扩展，每个字符最多保留 limit 次  

**关键点**:
```python
# 记录每个字符在栈中出现的次数
count_in_stack = defaultdict(int)

# 添加时检查次数
if count_in_stack[c] < limit:
    st.append(c)
    count_in_stack[c] += 1
```

---

### 5. [1081. 不同字符的最小子序列](https://leetcode.cn/problems/smallest-subsequence-of-distinct-characters/)
**难度**: 中等  
**题目难度**: 2185  
**标签**: 单调栈 + 去重  
**描述**: 与 316 题完全相同  
**提示**: 直接使用 316 题的解法

---

### 6. [321. 拼接最大数](https://leetcode.cn/problems/create-maximum-number/)
**难度**: 困难  
**标签**: 单调栈 + 归并  
**描述**: 从两个数组中分别选出数字，拼接成长度为 k 的最大数  
**重要性**: ⭐⭐⭐⭐ 综合性强，难度较大  

**思路分解**:
```python
# 步骤1: 从 nums1 选 i 个，nums2 选 k-i 个
# 使用类似 1673 的方法

def maxNumber(nums, k):
    """从一个数组选 k 个数，使其最大"""
    st = []
    n = len(nums)
    for i, x in enumerate(nums):
        while st and st[-1] < x and len(st) + (n-i) > k:
            st.pop()
        if len(st) < k:
            st.append(x)
    return st

# 步骤2: 合并两个序列
def merge(nums1, nums2):
    """合并两个序列，使结果最大"""
    ans = []
    while nums1 or nums2:
        if nums1 > nums2:  # 字典序比较
            ans.append(nums1.pop(0))
        else:
            ans.append(nums2.pop(0))
    return ans

# 步骤3: 枚举所有分配方案
result = []
for i in range(k + 1):
    if i <= len(nums1) and k-i <= len(nums2):
        sub1 = maxNumber(nums1, i)
        sub2 = maxNumber(nums2, k-i)
        merged = merge(sub1, sub2)
        result = max(result, merged)
```

**难点**:
- 如何从一个数组选 k 个数使其最大
- 如何合并两个序列使结果最大
- 字典序的比较

---

### 7. [2030. 含特定字母的最小子序列](https://leetcode.cn/problems/smallest-k-length-subsequence-with-occurrences-of-a-letter/)
**难度**: 困难  
**题目难度**: 2562  
**标签**: 单调栈 + 多重约束  
**描述**: 选择长度为 k 的子序列，使其字典序最小，且包含至少 repetition 个字母 letter  
**重要性**: ⭐⭐⭐⭐⭐ 最难的字典序题目  

**关键点**:
```python
# 需要同时满足多个约束：
# 1. 总长度为 k
# 2. 包含至少 repetition 个 letter

# 记录 letter 剩余次数
letter_left = s.count(letter)
# 记录已选的 letter 次数
letter_used = 0

for i, c in enumerate(s):
    # 删除栈顶的条件更复杂
    while st:
        top = st[-1]
        # 1. 后面还有足够元素
        if len(st) + (n - i) <= k:
            break
        # 2. 栈顶更大
        if top <= c:
            break
        # 3. 如果删除 letter，检查后面是否还够
        if top == letter:
            if letter_used + letter_left <= repetition:
                break
            letter_used -= 1
        
        st.pop()
    
    # 添加元素
    if len(st) < k:
        if c == letter:
            letter_used += 1
        st.append(c)
    
    if c == letter:
        letter_left -= 1
```

**难度分析**:
- 需要同时考虑三个约束条件
- 删除逻辑复杂，需要仔细分析
- 边界条件多，容易出错

---

## 📊 学习进度

### 必做题目
- [ ] 402. 移掉 K 位数字 ⭐⭐⭐⭐⭐
- [ ] 1673. 找出最具竞争力的子序列 ⭐⭐⭐⭐
- [ ] 316. 去除重复字母 ⭐⭐⭐⭐⭐
- [ ] 321. 拼接最大数 ⭐⭐⭐⭐

### 扩展题目
- [ ] 1081. 不同字符的最小子序列
- [ ] 316 扩展：重复个数不超过 limit
- [ ] 2030. 含特定字母的最小子序列 ⭐⭐⭐⭐⭐

## 💡 学习路线

### 第一阶段：理解删除和选择

#### 1. [402. 移掉 K 位数字](https://leetcode.cn/problems/remove-k-digits/) ⭐⭐⭐⭐⭐
**为什么先做这题？**
- 最简单的字典序问题
- 理解贪心策略：让高位尽可能小
- 学习边界处理：前导零、空字符串

**练习重点**:
- 什么时候删除栈顶？
- k 没用完怎么办？
- 如何处理前导零？

---

#### 2. [1673. 找出最具竞争力的子序列](https://leetcode.cn/problems/find-the-most-competitive-subsequence/) ⭐⭐⭐⭐
**为什么是第二题？**
- 与 402 互补：从删除到选择
- 理解"后面是否还够"的判断逻辑

**对比学习**:
```python
# 402: 删除 k 个，保留 n-k 个
可删除条件: k > 0

# 1673: 选择 k 个
可删除条件: len(st) + (n-i) > k
```

---

### 第二阶段：加入去重约束

#### 3. [316. 去除重复字母](https://leetcode.cn/problems/remove-duplicate-letters/) ⭐⭐⭐⭐⭐
**为什么第三做？**
- 在 402 基础上增加去重约束
- 学习使用辅助数据结构（`last`, `in_stack`）
- 理解新的删除条件：后面是否还会出现

**核心变化**:
```python
# 402/1673: 基于次数或长度判断
# 316: 基于"是否还会出现"判断

可删除条件: last[st[-1]] > i  # 后面还会出现
```

**技巧总结**:
- `last` 字典预处理最后位置
- `in_stack` 集合避免重复
- 跳过已在栈中的元素

---

### 第三阶段：综合应用

#### 4. [321. 拼接最大数](https://leetcode.cn/problems/create-maximum-number/) ⭐⭐⭐⭐
**为什么第四做？**
- 综合应用前面所学
- 分解成子问题：选择 + 合并
- 练习算法组合

**三个子问题**:
```python
1. 从一个数组选 k 个 → 使用 1673 的方法
2. 合并两个序列 → 归并 + 字典序比较
3. 枚举分配方案 → 遍历所有可能
```

**建议**:
- 先实现从一个数组选 k 个（类似 1673）
- 再实现合并两个序列
- 最后组合起来

---

### 第四阶段：挑战最难题

#### 5. [2030. 含特定字母的最小子序列](https://leetcode.cn/problems/smallest-k-length-subsequence-with-occurrences-of-a-letter/) ⭐⭐⭐⭐⭐
**为什么最后做？**
- 最复杂的约束条件
- 需要同时考虑长度和字母次数
- 删除逻辑最复杂

**三重约束**:
```python
1. 总长度 = k
2. 包含 >= repetition 个 letter
3. 字典序最小
```

**建议**:
- 在纸上画图推导
- 列出所有删除的条件
- 可以先看题解理解思路

---

## 🎯 解题模式总结

### 模式1: 删除型（402）
```python
特点: 删除 k 个元素
条件: k > 0
后处理: 
  - 如果 k 没用完，继续删除
  - 去除前导零（数字问题）
```

### 模式2: 选择型（1673）
```python
特点: 选择 k 个元素
条件: len(st) + (n - i) > k  # 后面还够
添加: if len(st) < k
```

### 模式3: 去重型（316）
```python
特点: 每个元素最多一次
辅助: 
  - last 字典
  - in_stack 集合
条件: 
  - 跳过已在栈中的
  - last[st[-1]] > i  # 后面还有
```

### 模式4: 合并型（321）
```python
特点: 从多个数组选择并合并
步骤:
  1. 分别从每个数组选择
  2. 合并序列（字典序比较）
  3. 枚举所有分配方案
```

### 模式5: 多重约束型（2030）
```python
特点: 多个约束同时满足
技巧:
  - 维护多个计数器
  - 逐个检查删除条件
  - 细心处理边界
```

## 🔍 调试技巧

### 1. 打印中间状态
```python
print(f"i={i}, c={c}, st={st}, k={k}")
```

### 2. 小数据测试
```python
# 手动模拟，检查每一步
num = "1432219", k = 3
期望: "1219"
```

### 3. 边界检查
```python
# 测试极端情况
- 全部相同: "111111", k=3
- 递增序列: "123456", k=3  
- 递减序列: "654321", k=3
- k=0, k=n
```

## 🎓 学习建议

1. **按顺序学习**: 402 → 1673 → 316 → 321 → 2030
2. **对比理解**: 402 和 1673 对比，理解删除 vs 选择
3. **画图模拟**: 复杂题目一定要画图
4. **总结模式**: 归纳不同题型的通用模式
5. **多次练习**: 字典序问题细节多，需要反复练习

## 🔗 相关资源

- 回到 [最小字典序专题](README.md)
- 回到 [单调栈主页](../README.md)
- 回到 [基础部分](../basics/README.md)

## 📚 推荐做题顺序

| 顺序 | 题目 | 难度 | 重要性 | 备注 |
|------|------|------|--------|------|
| 1 | 402 | 中等 | ⭐⭐⭐⭐⭐ | 入门必做 |
| 2 | 1673 | 中等 | ⭐⭐⭐⭐ | 与 402 对比 |
| 3 | 316 | 中等 | ⭐⭐⭐⭐⭐ | 加入去重 |
| 4 | 1081 | 中等 | ⭐⭐⭐ | 同 316 |
| 5 | 321 | 困难 | ⭐⭐⭐⭐ | 综合应用 |
| 6 | 2030 | 困难 | ⭐⭐⭐⭐⭐ | 最难挑战 |

---

**提示**: 字典序问题的关键在于"贪心 + 约束"，理解什么时候可以删除栈顶是核心！
