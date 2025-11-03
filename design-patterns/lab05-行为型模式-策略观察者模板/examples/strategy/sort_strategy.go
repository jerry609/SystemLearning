package main

import (
	"fmt"
	"math/rand"
	"time"
)

// 策略模式示例：排序策略
// 本示例展示了如何使用策略模式实现多种排序算法

// SortStrategy 排序策略接口
type SortStrategy interface {
	Sort(data []int) []int
	GetName() string
}

// BubbleSortStrategy 冒泡排序策略
type BubbleSortStrategy struct{}

func (b *BubbleSortStrategy) Sort(data []int) []int {
	result := make([]int, len(data))
	copy(result, data)

	n := len(result)
	for i := 0; i < n-1; i++ {
		swapped := false
		for j := 0; j < n-i-1; j++ {
			if result[j] > result[j+1] {
				result[j], result[j+1] = result[j+1], result[j]
				swapped = true
			}
		}
		// 如果没有发生交换，说明已经有序
		if !swapped {
			break
		}
	}
	return result
}

func (b *BubbleSortStrategy) GetName() string {
	return "冒泡排序"
}

// QuickSortStrategy 快速排序策略
type QuickSortStrategy struct{}

func (q *QuickSortStrategy) Sort(data []int) []int {
	result := make([]int, len(data))
	copy(result, data)
	q.quickSort(result, 0, len(result)-1)
	return result
}

func (q *QuickSortStrategy) quickSort(arr []int, low, high int) {
	if low < high {
		pi := q.partition(arr, low, high)
		q.quickSort(arr, low, pi-1)
		q.quickSort(arr, pi+1, high)
	}
}

func (q *QuickSortStrategy) partition(arr []int, low, high int) int {
	pivot := arr[high]
	i := low - 1

	for j := low; j < high; j++ {
		if arr[j] < pivot {
			i++
			arr[i], arr[j] = arr[j], arr[i]
		}
	}
	arr[i+1], arr[high] = arr[high], arr[i+1]
	return i + 1
}

func (q *QuickSortStrategy) GetName() string {
	return "快速排序"
}

// InsertionSortStrategy 插入排序策略
type InsertionSortStrategy struct{}

func (i *InsertionSortStrategy) Sort(data []int) []int {
	result := make([]int, len(data))
	copy(result, data)

	for j := 1; j < len(result); j++ {
		key := result[j]
		k := j - 1

		// 将大于 key 的元素向后移动
		for k >= 0 && result[k] > key {
			result[k+1] = result[k]
			k--
		}
		result[k+1] = key
	}
	return result
}

func (i *InsertionSortStrategy) GetName() string {
	return "插入排序"
}

// SelectionSortStrategy 选择排序策略
type SelectionSortStrategy struct{}

func (s *SelectionSortStrategy) Sort(data []int) []int {
	result := make([]int, len(data))
	copy(result, data)

	n := len(result)
	for i := 0; i < n-1; i++ {
		minIdx := i
		for j := i + 1; j < n; j++ {
			if result[j] < result[minIdx] {
				minIdx = j
			}
		}
		result[i], result[minIdx] = result[minIdx], result[i]
	}
	return result
}

func (s *SelectionSortStrategy) GetName() string {
	return "选择排序"
}

// MergeSortStrategy 归并排序策略
type MergeSortStrategy struct{}

func (m *MergeSortStrategy) Sort(data []int) []int {
	result := make([]int, len(data))
	copy(result, data)
	return m.mergeSort(result)
}

func (m *MergeSortStrategy) mergeSort(arr []int) []int {
	if len(arr) <= 1 {
		return arr
	}

	mid := len(arr) / 2
	left := m.mergeSort(arr[:mid])
	right := m.mergeSort(arr[mid:])

	return m.merge(left, right)
}

func (m *MergeSortStrategy) merge(left, right []int) []int {
	result := make([]int, 0, len(left)+len(right))
	i, j := 0, 0

	for i < len(left) && j < len(right) {
		if left[i] <= right[j] {
			result = append(result, left[i])
			i++
		} else {
			result = append(result, right[j])
			j++
		}
	}

	result = append(result, left[i:]...)
	result = append(result, right[j:]...)

	return result
}

func (m *MergeSortStrategy) GetName() string {
	return "归并排序"
}

// Sorter 排序器上下文
type Sorter struct {
	strategy SortStrategy
}

// NewSorter 创建排序器
func NewSorter(strategy SortStrategy) *Sorter {
	return &Sorter{
		strategy: strategy,
	}
}

// SetStrategy 设置排序策略
func (s *Sorter) SetStrategy(strategy SortStrategy) {
	s.strategy = strategy
}

// Sort 执行排序
func (s *Sorter) Sort(data []int) []int {
	if s.strategy == nil {
		return data
	}
	return s.strategy.Sort(data)
}

// GetStrategyName 获取当前策略名称
func (s *Sorter) GetStrategyName() string {
	if s.strategy == nil {
		return "未设置策略"
	}
	return s.strategy.GetName()
}

// SortWithTiming 执行排序并计时
func (s *Sorter) SortWithTiming(data []int) ([]int, time.Duration) {
	start := time.Now()
	result := s.Sort(data)
	duration := time.Since(start)
	return result, duration
}

// 辅助函数：生成随机数组
func generateRandomArray(size int, max int) []int {
	rand.Seed(time.Now().UnixNano())
	arr := make([]int, size)
	for i := 0; i < size; i++ {
		arr[i] = rand.Intn(max)
	}
	return arr
}

// 辅助函数：打印数组
func printArray(arr []int, limit int) {
	if len(arr) <= limit {
		fmt.Println(arr)
	} else {
		fmt.Printf("%v ... %v (共 %d 个元素)\n",
			arr[:limit/2], arr[len(arr)-limit/2:], len(arr))
	}
}

func main() {
	fmt.Println("=== 策略模式示例：排序策略 ===\n")

	// 测试数据
	smallData := []int{64, 34, 25, 12, 22, 11, 90, 88, 45, 50}
	fmt.Printf("原始数据: %v\n\n", smallData)

	// 创建排序器
	sorter := NewSorter(nil)

	// 测试不同的排序策略
	strategies := []SortStrategy{
		&BubbleSortStrategy{},
		&QuickSortStrategy{},
		&InsertionSortStrategy{},
		&SelectionSortStrategy{},
		&MergeSortStrategy{},
	}

	fmt.Println("【小数据集排序测试】")
	for _, strategy := range strategies {
		sorter.SetStrategy(strategy)
		result, duration := sorter.SortWithTiming(smallData)
		fmt.Printf("%-12s: %v (耗时: %v)\n",
			strategy.GetName(), result, duration)
	}

	// 性能对比测试
	fmt.Println("\n【性能对比测试】")
	sizes := []int{100, 1000, 5000}

	for _, size := range sizes {
		fmt.Printf("\n数据规模: %d 个元素\n", size)
		fmt.Println("----------------------------------------")

		largeData := generateRandomArray(size, 10000)

		for _, strategy := range strategies {
			sorter.SetStrategy(strategy)
			_, duration := sorter.SortWithTiming(largeData)
			fmt.Printf("%-12s: %v\n", strategy.GetName(), duration)
		}
	}

	// 展示策略的动态切换
	fmt.Println("\n【动态切换策略】")
	data1 := []int{5, 2, 8, 1, 9}
	data2 := []int{15, 12, 18, 11, 19, 13, 16}
	data3 := []int{25, 22, 28, 21, 29, 23, 26, 24, 27}

	fmt.Printf("\n数据集 1: %v\n", data1)
	sorter.SetStrategy(&BubbleSortStrategy{})
	fmt.Printf("使用 %s: %v\n", sorter.GetStrategyName(), sorter.Sort(data1))

	fmt.Printf("\n数据集 2: %v\n", data2)
	sorter.SetStrategy(&QuickSortStrategy{})
	fmt.Printf("使用 %s: %v\n", sorter.GetStrategyName(), sorter.Sort(data2))

	fmt.Printf("\n数据集 3: %v\n", data3)
	sorter.SetStrategy(&MergeSortStrategy{})
	fmt.Printf("使用 %s: %v\n", sorter.GetStrategyName(), sorter.Sort(data3))

	// 根据数据规模自动选择策略
	fmt.Println("\n【智能策略选择】")
	autoSelectAndSort := func(data []int) {
		fmt.Printf("数据: ")
		printArray(data, 10)

		var strategy SortStrategy
		size := len(data)

		if size < 10 {
			strategy = &InsertionSortStrategy{}
		} else if size < 100 {
			strategy = &QuickSortStrategy{}
		} else {
			strategy = &MergeSortStrategy{}
		}

		sorter.SetStrategy(strategy)
		result, duration := sorter.SortWithTiming(data)

		fmt.Printf("自动选择: %s\n", strategy.GetName())
		fmt.Printf("排序结果: ")
		printArray(result, 10)
		fmt.Printf("耗时: %v\n\n", duration)
	}

	autoSelectAndSort([]int{5, 2, 8, 1, 9})
	autoSelectAndSort(generateRandomArray(50, 100))
	autoSelectAndSort(generateRandomArray(500, 1000))

	fmt.Println("=== 示例结束 ===")
}
