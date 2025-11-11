// Go 二分查找模板package basicbinarysearch

package main

import (
	"fmt"
	"sort"
)

// LowerBound 返回 >= target 的第一个元素的下标
// 如果不存在，返回 len(nums)
func LowerBound(nums []int, target int) int {
	left, right := 0, len(nums)
	for left < right {
		mid := left + (right-left)/2
		if nums[mid] < target { // 红色区域
			left = mid + 1
		} else { // 蓝色区域，nums[mid] >= target
			right = mid
		}
	}
	return left
}

// LowerBoundClosed 闭区间写法
func LowerBoundClosed(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := left + (right-left)/2
		if nums[mid] < target {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return left
}

// FirstGreaterOrEqual 返回 >= target 的第一个元素的下标
func FirstGreaterOrEqual(nums []int, target int) int {
	return LowerBound(nums, target)
}

// FirstGreater 返回 > target 的第一个元素的下标
func FirstGreater(nums []int, target int) int {
	return LowerBound(nums, target+1)
}

// LastLess 返回 < target 的最后一个元素的下标
func LastLess(nums []int, target int) int {
	return LowerBound(nums, target) - 1
}

// LastLessOrEqual 返回 <= target 的最后一个元素的下标
func LastLessOrEqual(nums []int, target int) int {
	return LowerBound(nums, target+1) - 1
}

// CountLess 统计 < target 的元素个数
func CountLess(nums []int, target int) int {
	return LowerBound(nums, target)
}

// CountLessOrEqual 统计 <= target 的元素个数
func CountLessOrEqual(nums []int, target int) int {
	return LowerBound(nums, target+1)
}

// CountGreaterOrEqual 统计 >= target 的元素个数
func CountGreaterOrEqual(nums []int, target int) int {
	return len(nums) - LowerBound(nums, target)
}

// CountGreater 统计 > target 的元素个数
func CountGreater(nums []int, target int) int {
	return len(nums) - LowerBound(nums, target+1)
}

func main() {
	nums := []int{1, 3, 3, 5, 7, 7, 7, 9}
	target := 7

	fmt.Printf("数组: %v\n", nums)
	fmt.Printf("目标值: %d\n\n", target)

	// 查找位置
	fmt.Printf(">= %d 的第一个位置: %d\n", target, FirstGreaterOrEqual(nums, target))
	fmt.Printf("> %d 的第一个位置: %d\n", target, FirstGreater(nums, target))
	fmt.Printf("< %d 的最后一个位置: %d\n", target, LastLess(nums, target))
	fmt.Printf("<= %d 的最后一个位置: %d\n", target, LastLessOrEqual(nums, target))

	fmt.Println()

	// 统计个数
	fmt.Printf("< %d 的元素个数: %d\n", target, CountLess(nums, target))
	fmt.Printf("<= %d 的元素个数: %d\n", target, CountLessOrEqual(nums, target))
	fmt.Printf(">= %d 的元素个数: %d\n", target, CountGreaterOrEqual(nums, target))
	fmt.Printf("> %d 的元素个数: %d\n", target, CountGreater(nums, target))

	// 使用标准库
	fmt.Printf("\n使用 sort.SearchInts: %d\n", sort.SearchInts(nums, target))
}
