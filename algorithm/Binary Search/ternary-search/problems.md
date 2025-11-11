# 三、三分法 - 题目列表

## 题目列表

| 题号 | 题目 | 难度 | 难度分 | 说明 |
|------|------|------|--------|------|
| 1515 | [服务中心的最佳位置](https://leetcode.cn/problems/best-position-for-a-service-centre/) | 🔴 困难 | 2157 | **做法不止一种** |

## 题目详解

### ⭐ 1515. 服务中心的最佳位置

**题意**: 在二维平面上找一个点，使得到所有给定点的欧几里得距离之和最小。

**方法一：三分法（本专题）**

```python
def getMinDistSum(positions):
    def distance_sum(x, y):
        # 计算 (x, y) 到所有点的距离之和
        return sum(((x - px) ** 2 + (y - py) ** 2) ** 0.5 
                   for px, py in positions)
    
    # 确定搜索范围
    min_x = min(p[0] for p in positions)
    max_x = max(p[0] for p in positions)
    min_y = min(p[1] for p in positions)
    max_y = max(p[1] for p in positions)
    
    eps = 1e-7
    
    # 先对 x 坐标三分
    while max_x - min_x > eps:
        m1_x = min_x + (max_x - min_x) / 3
        m2_x = max_x - (max_x - min_x) / 3
        
        # 对每个 x，找最优的 y（再次三分）
        def min_dist_for_x(x):
            y_min, y_max = min_y, max_y
            while y_max - y_min > eps:
                m1_y = y_min + (y_max - y_min) / 3
                m2_y = y_max - (y_max - y_min) / 3
                
                if distance_sum(x, m1_y) > distance_sum(x, m2_y):
                    y_min = m1_y
                else:
                    y_max = m2_y
            return distance_sum(x, (y_min + y_max) / 2)
        
        if min_dist_for_x(m1_x) > min_dist_for_x(m2_x):
            min_x = m1_x
        else:
            max_x = m2_x
    
    # 最终位置
    best_x = (min_x + max_x) / 2
    
    # 对 y 再三分一次
    y_min, y_max = min_y, max_y
    while y_max - y_min > eps:
        m1_y = y_min + (y_max - y_min) / 3
        m2_y = y_max - (y_max - y_min) / 3
        
        if distance_sum(best_x, m1_y) > distance_sum(best_x, m2_y):
            y_min = m1_y
        else:
            y_max = m2_y
    
    return distance_sum(best_x, (y_min + y_max) / 2)
```

**方法二：梯度下降**

```python
def getMinDistSum(positions):
    # 初始位置：所有点的重心
    x = sum(p[0] for p in positions) / len(positions)
    y = sum(p[1] for p in positions) / len(positions)
    
    learning_rate = 1.0
    eps = 1e-7
    
    while learning_rate > eps:
        # 计算梯度
        grad_x, grad_y = 0, 0
        
        for px, py in positions:
            dist = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
            if dist > eps:
                grad_x += (x - px) / dist
                grad_y += (y - py) / dist
        
        # 更新位置
        new_x = x - learning_rate * grad_x
        new_y = y - learning_rate * grad_y
        
        # 如果没有改进，减小学习率
        def distance_sum(x, y):
            return sum(((x - px) ** 2 + (y - py) ** 2) ** 0.5 
                      for px, py in positions)
        
        if distance_sum(new_x, new_y) < distance_sum(x, y):
            x, y = new_x, new_y
        else:
            learning_rate *= 0.5
    
    return sum(((x - px) ** 2 + (y - py) ** 2) ** 0.5 
               for px, py in positions)
```

**方法三：模拟退火**

```python
import random
import math

def getMinDistSum(positions):
    def distance_sum(x, y):
        return sum(((x - px) ** 2 + (y - py) ** 2) ** 0.5 
                   for px, py in positions)
    
    # 初始位置
    x = sum(p[0] for p in positions) / len(positions)
    y = sum(p[1] for p in positions) / len(positions)
    
    temperature = 100
    cooling_rate = 0.99
    min_temp = 1e-8
    
    best_dist = distance_sum(x, y)
    
    while temperature > min_temp:
        # 随机扰动
        new_x = x + random.uniform(-1, 1) * temperature
        new_y = y + random.uniform(-1, 1) * temperature
        
        new_dist = distance_sum(new_x, new_y)
        delta = new_dist - best_dist
        
        # Metropolis 准则
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            x, y = new_x, new_y
            best_dist = new_dist
        
        temperature *= cooling_rate
    
    return best_dist
```

## 算法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| 三分法 | 理论保证，精确 | 需要二维三分，较慢 | 小数据量 |
| 梯度下降 | 快速收敛 | 可能陷入局部最优 | 凸优化问题 |
| 模拟退火 | 跳出局部最优 | 不稳定，参数敏感 | 非凸问题 |

## 关键点

### 1. 二维三分

对于二维问题：
- 可以对 x 和 y 分别三分
- 也可以固定一维，三分另一维

### 2. 精度控制

```python
eps = 1e-7  # 根据题目要求设置
while right - left > eps:
    # ...
```

### 3. 单峰性判断

这道题的距离和函数是凸函数（单谷），所以可以用三分法。

## 为什么距离和是凸函数？

对于欧几里得距离和：
$$f(x, y) = \sum_{i=1}^{n} \sqrt{(x - x_i)^2 + (y - y_i)^2}$$

这是一个凸函数，因为：
- 每个 $\sqrt{(x - x_i)^2 + (y - y_i)^2}$ 都是凸函数
- 凸函数的非负线性组合仍是凸函数

## 练习建议

1. **理解三分法原理**：为什么能找到极值
2. **对比其他方法**：梯度下降、模拟退火
3. **注意精度**：浮点运算的精度控制
4. **判断单峰性**：不是所有问题都能用三分

## 扩展知识

### 黄金分割搜索

比例取 $\phi = \frac{\sqrt{5} - 1}{2} \approx 0.618$：

```python
phi = (5 ** 0.5 - 1) / 2

def golden_section_search(left, right, eps=1e-6):
    m1 = right - (right - left) * phi
    m2 = left + (right - left) * phi
    
    f1, f2 = f(m1), f(m2)
    
    while right - left > eps:
        if f1 < f2:
            right = m2
            m2 = m1
            f2 = f1
            m1 = right - (right - left) * phi
            f1 = f(m1)
        else:
            left = m1
            m1 = m2
            f1 = f2
            m2 = left + (right - left) * phi
            f2 = f(m2)
    
    return (left + right) / 2
```

**优点**: 每次只需计算一个新的函数值（复用之前的值）

---

**返回**: [三分法](README.md) | [二分查找专题](../README.md)
