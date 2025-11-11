package monotonic_stack

// nearestGreater 返回数组中每个元素左侧和右侧最近的严格大于它的元素的下标
// 时间复杂度: O(n)
// 空间复杂度: O(n)
func nearestGreater(nums []int) ([]int, []int) {
	n := len(nums)

	// left[i] 是 nums[i] 左侧最近的严格大于 nums[i] 的数的下标，若不存在则为 -1
	left := make([]int, n)
	st := []int{-1} // 哨兵
	for i, x := range nums {
		for len(st) > 1 && nums[st[len(st)-1]] <= x { // 如果求严格小于，改成 >=
			st = st[:len(st)-1]
		}
		left[i] = st[len(st)-1]
		st = append(st, i)
	}

	// right[i] 是 nums[i] 右侧最近的严格大于 nums[i] 的数的下标，若不存在则为 n
	right := make([]int, n)
	st = []int{n} // 哨兵
	for i := n - 1; i >= 0; i-- {
		x := nums[i]
		for len(st) > 1 && nums[st[len(st)-1]] <= x {
			st = st[:len(st)-1]
		}
		right[i] = st[len(st)-1]
		st = append(st, i)
	}

	return left, right
}

// nearestSmaller 返回数组中每个元素左侧和右侧最近的严格小于它的元素的下标
func nearestSmaller(nums []int) ([]int, []int) {
	n := len(nums)

	left := make([]int, n)
	st := []int{-1}
	for i, x := range nums {
		for len(st) > 1 && nums[st[len(st)-1]] >= x { // 改成 >= 就是求严格小于
			st = st[:len(st)-1]
		}
		left[i] = st[len(st)-1]
		st = append(st, i)
	}

	right := make([]int, n)
	st = []int{n}
	for i := n - 1; i >= 0; i-- {
		x := nums[i]
		for len(st) > 1 && nums[st[len(st)-1]] >= x {
			st = st[:len(st)-1]
		}
		right[i] = st[len(st)-1]
		st = append(st, i)
	}

	return left, right
}
