# §2.1 求最小

二分答案求最小值，本质是在虚拟数组中找第一个满足条件的位置。

## 核心特点

- **红蓝分布**: 左边不满足（红色），右边满足（蓝色）
- **更新策略**: `check(mid) == true` 时更新 `right = mid`
- **返回值**: 返回 `right`（check 更新谁就返回谁）

## 开区间二分模板（求最小）

```python
class Solution:
    # 计算满足 check(x) == True 的最小整数 x
    def binarySearchMin(self, nums: List[int]) -> int:
        # 二分猜答案：判断 mid 是否满足题目要求
        def check(mid: int) -> bool:
            # TODO: 实现检查逻辑
            pass

        left =   # 循环不变量：check(left) 恒为 False
        right =  # 循环不变量：check(right) 恒为 True
        
        while left + 1 < right:  # 开区间不为空
            mid = (left + right) // 2
            if check(mid):  # 说明 check(>= mid 的数) 均为 True
                right = mid  # 接下来在 (left, mid) 中二分答案
            else:  # 说明 check(<= mid 的数) 均为 False
                left = mid  # 接下来在 (mid, right) 中二分答案
        
        # 循环结束后 left+1 = right
        # 此时 check(left) == False 而 check(left+1) == check(right) == True
        # 所以 right 就是最小的满足 check 的值
        return right
```

## 题目特征

求最小值的题目通常有以下特征：
- 要求"最少"、"最小"
- 存在单调性：答案越大越容易满足条件
- 需要找临界值

## 学习建议

1. 先理解循环不变量
2. 注意 check 函数的编写
3. 理解为什么返回 right
4. 对比闭区间写法的区别

---

**返回**: [二分答案](../README.md) | [二分查找专题](../../README.md)
