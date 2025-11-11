# §0.3 遍历对角线题目列表

## 📝 核心题目

### 1. [3446. 按对角线进行矩阵排序](https://leetcode.cn/problems/sort-the-matrix-diagonally/)
**难度**: 中等  
**题目难度**: 1373  
**标签**: 对角线排序  
**描述**: 对矩阵的每条对角线分别排序  
**重要性**: ⭐⭐⭐⭐ 对角线遍历的入门题  

**解法**:
```python
def diagonalSort(mat: List[List[int]]) -> List[List[int]]:
    m, n = len(mat), len(mat[0])
    
    # 按主对角线分组
    diag_dict = {}
    for i in range(m):
        for j in range(n):
            key = i - j  # 对角线编号
            if key not in diag_dict:
                diag_dict[key] = []
            diag_dict[key].append(mat[i][j])
    
    # 对每条对角线排序
    for key in diag_dict:
        diag_dict[key].sort()
    
    # 写回矩阵
    diag_idx = {key: 0 for key in diag_dict}
    for i in range(m):
        for j in range(n):
            key = i - j
            mat[i][j] = diag_dict[key][diag_idx[key]]
            diag_idx[key] += 1
    
    return mat
```

---

### 2. [2711. 对角线上不同值的数量差](https://leetcode.cn/problems/difference-of-number-of-distinct-values-on-diagonals/)
**难度**: 中等  
**题目难度**: 1429  
**标签**: 对角线统计  
**描述**: 计算每个位置左上和右下对角线上不同值的数量差  

**解法思路**:
```python
# 对于每个位置 (i, j)
# 统计左上对角线（i-j 相同）中比 (i,j) 小的位置的不同值
# 统计右下对角线（i-j 相同）中比 (i,j) 大的位置的不同值
# 计算差值
```

---

### 3. [1329. 将矩阵按对角线排序](https://leetcode.cn/problems/sort-the-matrix-diagonally/)
**难度**: 中等  
**题目难度**: 1548  
**标签**: 对角线排序  
**描述**: 对矩阵的每条对角线从左上到右下排序  
**提示**: 同 3446 题，可能是同一题的不同链接  

---

### 4. [498. 对角线遍历](https://leetcode.cn/problems/diagonal-traverse/)
**难度**: 中等  
**标签**: 对角线遍历顺序  
**描述**: 按照之字形顺序返回矩阵的所有元素  
**重要性**: ⭐⭐⭐⭐ 理解对角线遍历的方向切换  

**解法**:
```python
def findDiagonalOrder(mat: List[List[int]]) -> List[int]:
    m, n = len(mat), len(mat[0])
    result = []
    
    # 遍历所有副对角线（i + j = k）
    for k in range(m + n - 1):
        diagonal = []
        
        # 确定起点
        i = max(0, k - n + 1)
        j = min(k, n - 1)
        
        # 收集对角线元素
        while i < m and j >= 0:
            diagonal.append(mat[i][j])
            i += 1
            j -= 1
        
        # 奇数对角线需要反转
        if k % 2 == 0:
            diagonal.reverse()
        
        result.extend(diagonal)
    
    return result
```

---

### 5. [面试题 17.23. 最大黑方阵](https://leetcode.cn/problems/max-black-square-lcci/)
**难度**: 困难  
**标签**: 对角线 + 优化  
**描述**: 找到全黑边框的最大正方形  
**要求**: 做到 O(n² log n)，难度约 2800  

**优化思路**:
- 预处理每个位置向右和向下的连续 1 的个数
- 枚举正方形的左上角和边长
- 利用预处理信息 O(1) 判断是否合法

---

### 6. [562. 矩阵中最长的连续1线段](https://leetcode.cn/problems/longest-line-of-consecutive-one-in-matrix/)
**难度**: 中等  
**标签**: 四个方向  
**会员题**  
**描述**: 找到矩阵中最长的连续 1 线段（水平、垂直、对角线、反对角线）  

**解法思路**:
```python
# 对于每个方向，使用动态规划
# dp[i][j][d] 表示以 (i,j) 结尾，方向 d 的最长连续 1

# 方向：
# 0: 水平 →
# 1: 垂直 ↓
# 2: 主对角线 ↘
# 3: 副对角线 ↙

for i in range(m):
    for j in range(n):
        if matrix[i][j] == 1:
            dp[i][j][0] = dp[i][j-1][0] + 1 if j > 0 else 1
            dp[i][j][1] = dp[i-1][j][1] + 1 if i > 0 else 1
            dp[i][j][2] = dp[i-1][j-1][2] + 1 if i > 0 and j > 0 else 1
            dp[i][j][3] = dp[i-1][j+1][3] + 1 if i > 0 and j < n-1 else 1
```

---

## 📊 学习进度

### 入门题
- [ ] 3446. 按对角线进行矩阵排序 ⭐⭐⭐⭐
- [ ] 1329. 将矩阵按对角线排序

### 统计问题
- [ ] 2711. 对角线上不同值的数量差

### 遍历顺序
- [ ] 498. 对角线遍历 ⭐⭐⭐⭐

### 高级应用
- [ ] 面试题 17.23. 最大黑方阵
- [ ] 562. 矩阵中最长的连续1线段（会员题）

## 💡 解题技巧

### 技巧1: 对角线编号

**主对角线（↘）**:
```python
# 同一对角线：i - j = k
key = i - j
# k 范围：[-(n-1), m-1]
```

**副对角线（↙）**:
```python
# 同一对角线：i + j = k
key = i + j
# k 范围：[0, m+n-2]
```

### 技巧2: 哈希表分组

```python
# 按对角线分组
diag_dict = {}
for i in range(m):
    for j in range(n):
        key = i - j  # 或 i + j
        if key not in diag_dict:
            diag_dict[key] = []
        diag_dict[key].append(matrix[i][j])

# 对每条对角线处理
for key, diagonal in diag_dict.items():
    process(diagonal)
```

### 技巧3: 直接遍历

```python
# 遍历所有主对角线
for k in range(-(n-1), m):
    i = max(0, k)
    j = max(0, -k)
    
    while i < m and j < n:
        process(matrix[i][j])
        i += 1
        j += 1
```

### 技巧4: 动态规划（四个方向）

```python
# 对于需要统计四个方向的问题
directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # →, ↓, ↘, ↙
dp = [[[0] * 4 for _ in range(n)] for _ in range(m)]

for i in range(m):
    for j in range(n):
        if matrix[i][j] == 1:
            for d, (di, dj) in enumerate(directions):
                pi, pj = i - di, j - dj
                if 0 <= pi < m and 0 <= pj < n:
                    dp[i][j][d] = dp[pi][pj][d] + 1
                else:
                    dp[i][j][d] = 1
```

## 🎯 学习路线

### 第一步：理解对角线概念
**[3446. 按对角线进行矩阵排序](https://leetcode.cn/problems/sort-the-matrix-diagonally/)** ⭐⭐⭐⭐

**学习要点**:
1. 理解 `i - j` 表示主对角线
2. 掌握哈希表分组的方法
3. 学会写回矩阵

---

### 第二步：掌握遍历顺序
**[498. 对角线遍历](https://leetcode.cn/problems/diagonal-traverse/)** ⭐⭐⭐⭐

**学习要点**:
1. 理解之字形遍历
2. 掌握方向切换的规律
3. 处理边界情况

---

### 第三步：应用到统计问题
**[2711. 对角线上不同值的数量差](https://leetcode.cn/problems/difference-of-number-of-distinct-values-on-diagonals/)**

**学习要点**:
1. 在对角线上统计信息
2. 分别处理左上和右下部分

---

### 第四步：多方向问题
**[562. 矩阵中最长的连续1线段](https://leetcode.cn/problems/longest-line-of-consecutive-one-in-matrix/)**（会员题）

**学习要点**:
1. 同时处理四个方向
2. 使用三维 DP
3. 理解不同方向的转移

---

### 第五步：挑战高难度
**[面试题 17.23. 最大黑方阵](https://leetcode.cn/problems/max-black-square-lcci/)**

**学习要点**:
1. 预处理优化
2. 复杂度优化到 O(n² log n)
3. 巧妙利用对角线性质

## 🔍 常见问题

### Q1: 主对角线和副对角线有什么区别？
**A**: 
- 主对角线（↘）：`i - j = 常数`，从左上到右下
- 副对角线（↙）：`i + j = 常数`，从右上到左下

### Q2: 如何确定对角线的起点？
**A**: 
```python
# 主对角线
if k >= 0:
    start = (k, 0)  # 从左边界
else:
    start = (0, -k)  # 从上边界

# 副对角线
i_start = max(0, k - n + 1)
j_start = min(k, n - 1)
```

### Q3: 直接遍历好还是哈希表分组好？
**A**: 
- 直接遍历：空间 O(1)，适合只需要遍历一次的情况
- 哈希表分组：空间 O(mn)，适合需要多次访问或排序的情况

### Q4: 对角线遍历的方向怎么控制？
**A**: 
```python
# 之字形遍历
if k % 2 == 0:
    diagonal.reverse()  # 偶数对角线反转
```

## 🔗 相关资源

- 回到 [§0.3 遍历对角线](README.md)
- 回到 [枚举专题主页](../README.md)

## 📚 推荐学习顺序

| 顺序 | 题目 | 难度 | 重要性 | 备注 |
|------|------|------|--------|------|
| 1 | 3446 | 中等 | ⭐⭐⭐⭐ | 对角线入门 |
| 2 | 498 | 中等 | ⭐⭐⭐⭐ | 遍历顺序 |
| 3 | 2711 | 中等 | ⭐⭐⭐ | 统计应用 |
| 4 | 562 | 中等 | ⭐⭐⭐⭐ | 多方向DP（会员） |
| 5 | 面试题17.23 | 困难 | ⭐⭐⭐⭐⭐ | 优化挑战 |
