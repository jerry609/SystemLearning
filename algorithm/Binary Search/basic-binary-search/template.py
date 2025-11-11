# Python 二分查找模板

from typing import List
from bisect import bisect_left, bisect_right


class BinarySearchTemplate:
    """
    二分查找模板类
    基于红蓝染色法的实现
    """
    
    @staticmethod
    def lowerBound(nums: List[int], target: int) -> int:
        """
        返回 >= target 的第一个元素的下标
        如果不存在，返回 len(nums)
        
        等价于 bisect.bisect_left(nums, target)
        """
        left, right = 0, len(nums)
        while left < right:
            mid = (left + right) // 2
            if nums[mid] < target:  # 红色区域
                left = mid + 1
            else:  # 蓝色区域，nums[mid] >= target
                right = mid
        return left
    
    @staticmethod
    def lowerBound_closed(nums: List[int], target: int) -> int:
        """
        闭区间写法
        返回 >= target 的第一个元素的下标
        """
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return left
    
    @staticmethod
    def firstGreaterOrEqual(nums: List[int], target: int) -> int:
        """返回 >= target 的第一个元素的下标"""
        return BinarySearchTemplate.lowerBound(nums, target)
    
    @staticmethod
    def firstGreater(nums: List[int], target: int) -> int:
        """返回 > target 的第一个元素的下标"""
        return BinarySearchTemplate.lowerBound(nums, target + 1)
    
    @staticmethod
    def lastLess(nums: List[int], target: int) -> int:
        """返回 < target 的最后一个元素的下标"""
        return BinarySearchTemplate.lowerBound(nums, target) - 1
    
    @staticmethod
    def lastLessOrEqual(nums: List[int], target: int) -> int:
        """返回 <= target 的最后一个元素的下标"""
        return BinarySearchTemplate.lowerBound(nums, target + 1) - 1
    
    @staticmethod
    def countLess(nums: List[int], target: int) -> int:
        """统计 < target 的元素个数"""
        return BinarySearchTemplate.lowerBound(nums, target)
    
    @staticmethod
    def countLessOrEqual(nums: List[int], target: int) -> int:
        """统计 <= target 的元素个数"""
        return BinarySearchTemplate.lowerBound(nums, target + 1)
    
    @staticmethod
    def countGreaterOrEqual(nums: List[int], target: int) -> int:
        """统计 >= target 的元素个数"""
        return len(nums) - BinarySearchTemplate.lowerBound(nums, target)
    
    @staticmethod
    def countGreater(nums: List[int], target: int) -> int:
        """统计 > target 的元素个数"""
        return len(nums) - BinarySearchTemplate.lowerBound(nums, target + 1)


# 使用示例
if __name__ == "__main__":
    nums = [1, 3, 3, 5, 7, 7, 7, 9]
    target = 7
    
    print(f"数组: {nums}")
    print(f"目标值: {target}\n")
    
    # 查找位置
    print(f">= {target} 的第一个位置: {BinarySearchTemplate.firstGreaterOrEqual(nums, target)}")
    print(f"> {target} 的第一个位置: {BinarySearchTemplate.firstGreater(nums, target)}")
    print(f"< {target} 的最后一个位置: {BinarySearchTemplate.lastLess(nums, target)}")
    print(f"<= {target} 的最后一个位置: {BinarySearchTemplate.lastLessOrEqual(nums, target)}")
    
    print()
    
    # 统计个数
    print(f"< {target} 的元素个数: {BinarySearchTemplate.countLess(nums, target)}")
    print(f"<= {target} 的元素个数: {BinarySearchTemplate.countLessOrEqual(nums, target)}")
    print(f">= {target} 的元素个数: {BinarySearchTemplate.countGreaterOrEqual(nums, target)}")
    print(f"> {target} 的元素个数: {BinarySearchTemplate.countGreater(nums, target)}")
    
    # 使用内置函数
    print(f"\n使用 bisect_left: {bisect_left(nums, target)}")
    print(f"使用 bisect_right: {bisect_right(nums, target)}")
