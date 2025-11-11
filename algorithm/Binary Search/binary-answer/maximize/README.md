# §2.2 求最大

二分答案求最大值，本质是在虚拟数组中找最后一个满足条件的位置。

## 核心特点

- **红蓝分布**: 左边满足（蓝色），右边不满足（红色）
- **更新策略**: `check(mid) == true` 时更新 `left = mid`
- **返回值**: 返回 `left`（check 更新谁就返回谁）

## 对比求最小

| 特征 | 求最小 | 求最大 |
|------|--------|--------|
| check 为 true 时 | `right = mid` | `left = mid` ⚠️ |
| 返回值 | `right` | `left` ⚠️ |
| 类比 | 找第一个满足的 | 找最后一个满足的 |

**记忆方法**: 开区间写法中，check 更新的是谁，最终就返回谁。

## 开区间二分模板（求最大）

```python
class Solution:
    # 计算满足 check(x) == True 的最大整数 x
    def binarySearchMax(self, nums: List[int]) -> int:
        # 二分猜答案：判断 mid 是否满足题目要求
        def check(mid: int) -> bool:
            # TODO: 实现检查逻辑
            pass

        left =   # 循环不变量：check(left) 恒为 True
        right =  # 循环不变量：check(right) 恒为 False
        
        while left + 1 < right:
            mid = (left + right) // 2
            if check(mid):
                left = mid  # 注意这里更新的是 left，和求最小反过来
            else:
                right = mid
        
        # 循环结束后 left+1 = right
        # 此时 check(left) == True 而 check(left+1) == check(right) == False
        # 所以 left 就是最大的满足 check 的值
        return left  # check 更新的是谁，最终就返回谁
```

## 题目特征

求最大值的题目通常有以下特征：
- 要求"最多"、"最大"
- 存在单调性：答案越小越容易满足条件
- 需要找临界值

## 重要提示

⚠️ **注意求最大和求最小的写法区别**：
- 求最小：check 为 true 时缩小右边界 `right = mid`，返回 `right`
- 求最大：check 为 true 时扩大左边界 `left = mid`，返回 `left`

这是二分答案中最容易出错的地方！

## 学习建议

1. 对比求最小的写法，理解差异
2. 记住"check 更新谁就返回谁"
3. 注意循环不变量的变化
4. 多做题目体会差异

---

**返回**: [二分答案](../README.md) | [二分查找专题](../../README.md)
