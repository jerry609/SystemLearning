# §1.2 进阶题目列表（选做）

## 📝 标准进阶题目

### 1. [1019. 链表中的下一个更大节点](https://leetcode.cn/problems/next-greater-node-in-linked-list/)
**难度**: 中等  
**题目难度**: 1571  
**标签**: 单调栈 + 链表  
**描述**: 在链表中找到每个节点的下一个更大值

---

### 2. [768. 最多能完成排序的块 II](https://leetcode.cn/problems/max-chunks-to-make-sorted-ii/)
**难度**: 困难  
**题目难度**: 1788  
**标签**: 单调栈 + 分块  
**描述**: 将数组分成若干块，使得分别排序后能得到有序数组

---

### 3. [654. 最大二叉树](https://leetcode.cn/problems/maximum-binary-tree/)
**难度**: 中等  
**标签**: 单调栈 + 树构建  
**描述**: 根据最大值构建二叉树，要求做到 O(n)  
**提示**: 使用单调栈优化，避免递归的 O(n²) 复杂度

---

### 4. [456. 132 模式](https://leetcode.cn/problems/132-pattern/)
**难度**: 中等  
**题目难度**: 约 2000  
**标签**: 单调栈 + 模式匹配  
**描述**: 找到满足 i < j < k 且 nums[i] < nums[k] < nums[j] 的三元组

---

### 5. [3113. 边界元素是最大值的子数组数目](https://leetcode.cn/problems/find-the-number-of-subarrays-where-boundary-elements-are-maximum/)
**难度**: 困难  
**题目难度**: 2046  
**标签**: 单调栈 + 计数  
**描述**: 统计满足边界元素是子数组最大值的子数组数量

---

### 6. [2866. 美丽塔 II](https://leetcode.cn/problems/beautiful-towers-ii/)
**难度**: 中等  
**题目难度**: 2072  
**标签**: 单调栈 + 贪心  
**描述**: 构建美丽塔，使得高度满足特定条件

---

### 7. [1944. 队列中可以看到的人数](https://leetcode.cn/problems/number-of-visible-people-in-a-queue/)
**难度**: 困难  
**题目难度**: 2105  
**标签**: 单调栈 + 视线问题  
**描述**: 计算队列中每个人能看到的人数

---

### 8. [2454. 下一个更大元素 IV](https://leetcode.cn/problems/next-greater-element-iv/)
**难度**: 困难  
**题目难度**: 2175  
**标签**: 单调栈 + 双栈  
**描述**: 找到第二个更大的元素

---

### 9. [1130. 叶值的最小代价生成树](https://leetcode.cn/problems/minimum-cost-tree-from-leaf-values/)
**难度**: 中等  
**标签**: 单调栈 + DP优化  
**描述**: 使用 O(n) 做法构建最小代价生成树  
**提示**: 单调栈优化动态规划

---

### 10. [2289. 使数组按非递减顺序排列](https://leetcode.cn/problems/steps-to-make-array-non-decreasing/)
**难度**: 困难  
**题目难度**: 2482  
**标签**: 单调栈 + 模拟  
**描述**: 计算使数组非递减所需的步数

---

### 11. [1776. 车队 II](https://leetcode.cn/problems/car-fleet-ii/)
**难度**: 困难  
**题目难度**: 2531  
**标签**: 单调栈 + 物理模拟  
**描述**: 计算车队碰撞时间

---

### 12. [2736. 最大和查询](https://leetcode.cn/problems/maximum-sum-queries/)
**难度**: 困难  
**题目难度**: 2533  
**标签**: 单调栈 + 查询优化  
**描述**: 处理最大和查询问题

---

### 13. [3420. 统计 K 次操作以内得到非递减子数组的数目](https://leetcode.cn/problems/count-non-decreasing-subarrays-after-k-operations/)
**难度**: 困难  
**题目难度**: 2855  
**标签**: 单调栈 + 树形结构  
**描述**: 结合树形结构统计满足条件的子数组

---

## 📝 会员题目

### 14. [3221. 最大数组跳跃得分 II](https://leetcode.cn/problems/maximum-array-hopping-score-ii/)
**难度**: 中等  
**标签**: 单调栈 + 动态规划  
**会员题**

---

### 15. [1966. 未排序数组中的可被二分搜索的数](https://leetcode.cn/problems/binary-searchable-numbers-in-an-unsorted-array/)
**难度**: 中等  
**标签**: 单调栈 + 二分性质  
**会员题**

---

### 16. [2832. 每个元素为最大值的最大范围](https://leetcode.cn/problems/maximal-range-that-each-element-is-maximum-in-it/)
**难度**: 中等  
**标签**: 单调栈基础应用  
**会员题**

---

### 17. [2282. 在一个网格中可以看到的人数](https://leetcode.cn/problems/number-of-people-that-can-be-seen-in-a-grid/)
**难度**: 困难  
**标签**: 单调栈 + 二维问题  
**会员题**

---

### 18. [3555. 排序每个滑动窗口中最小的子数组](https://leetcode.cn/problems/find-the-count-of-good-integers/)
**难度**: 困难  
**标签**: 单调栈 + 滑动窗口  
**会员题** - 非暴力做法

---

## 🌟 思维扩展题目

### 19. [962. 最大宽度坡](https://leetcode.cn/problems/maximum-width-ramp/)
**难度**: 中等  
**题目难度**: 1608  
**标签**: 单调栈 + 贪心  
**描述**: 找到最大的 j - i，使得 nums[i] <= nums[j]

---

### 20. [3542. 将所有元素变为 0 的最少操作次数](https://leetcode.cn/problems/maximum-number-of-moves-to-kill-all-pawns/)
**难度**: 困难  
**题目难度**: 1890  
**标签**: 单调栈 + 操作优化  
**描述**: 计算最少操作次数

---

### 21. [1124. 表现良好的最长时间段](https://leetcode.cn/problems/longest-well-performing-interval/)
**难度**: 中等  
**题目难度**: 1908  
**标签**: 单调栈 + 前缀和  
**描述**: 找到最长的表现良好时间段

---

## 📊 学习进度

### 标准题目 (1-13)
- [ ] 1019. 链表中的下一个更大节点
- [ ] 768. 最多能完成排序的块 II
- [ ] 654. 最大二叉树
- [ ] 456. 132 模式
- [ ] 3113. 边界元素是最大值的子数组数目
- [ ] 2866. 美丽塔 II
- [ ] 1944. 队列中可以看到的人数
- [ ] 2454. 下一个更大元素 IV
- [ ] 1130. 叶值的最小代价生成树
- [ ] 2289. 使数组按非递减顺序排列
- [ ] 1776. 车队 II
- [ ] 2736. 最大和查询
- [ ] 3420. 统计 K 次操作以内得到非递减子数组的数目

### 会员题目 (14-18)
- [ ] 3221. 最大数组跳跃得分 II
- [ ] 1966. 未排序数组中的可被二分搜索的数
- [ ] 2832. 每个元素为最大值的最大范围
- [ ] 2282. 在一个网格中可以看到的人数
- [ ] 3555. 排序每个滑动窗口中最小的子数组

### 思维扩展 (19-21)
- [ ] 962. 最大宽度坡
- [ ] 3542. 将所有元素变为 0 的最少操作次数
- [ ] 1124. 表现良好的最长时间段

## 💡 学习建议

1. **循序渐进**: 按难度分从低到高练习
2. **时间管理**: 遇到困难题目，可以先看提示再思考
3. **总结归纳**: 相似题目要总结共同模式
4. **优化意识**: 注重时间复杂度的优化，从 O(n²) 到 O(n)
5. **会员题可选**: 如果没有 LeetCode 会员，可以跳过会员题

## 🔗 相关资源

- 回到 [§1.2 进阶](README.md)
- 回到 [基础部分](../basics/README.md)
- 继续学习 [矩形专题](../rectangle/README.md)
