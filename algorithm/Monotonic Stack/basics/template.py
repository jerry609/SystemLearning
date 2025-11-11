from typing import List, Tuple


def nearest_greater(nums: List[int]) -> Tuple[List[int], List[int]]:
    """
    返回数组中每个元素左侧和右侧最近的严格大于它的元素的下标
    
    时间复杂度: O(n)
    空间复杂度: O(n)
    
    Args:
        nums: 输入数组
        
    Returns:
        (left, right): 
            left[i] - nums[i] 左侧最近的严格大于 nums[i] 的数的下标，不存在则为 -1
            right[i] - nums[i] 右侧最近的严格大于 nums[i] 的数的下标，不存在则为 n
    """
    n = len(nums)
    
    # left[i] 是 nums[i] 左侧最近的严格大于 nums[i] 的数的下标，若不存在则为 -1
    left = [0] * n
    st = [-1]  # 哨兵
    for i, x in enumerate(nums):
        while len(st) > 1 and nums[st[-1]] <= x:  # 如果求严格小于，改成 >=
            st.pop()
        left[i] = st[-1]
        st.append(i)
    
    # right[i] 是 nums[i] 右侧最近的严格大于 nums[i] 的数的下标，若不存在则为 n
    right = [0] * n
    st = [n]  # 哨兵
    for i in range(n - 1, -1, -1):
        x = nums[i]
        while len(st) > 1 and nums[st[-1]] <= x:
            st.pop()
        right[i] = st[-1]
        st.append(i)
    
    return left, right


def nearest_smaller(nums: List[int]) -> Tuple[List[int], List[int]]:
    """
    返回数组中每个元素左侧和右侧最近的严格小于它的元素的下标
    """
    n = len(nums)
    
    left = [0] * n
    st = [-1]
    for i, x in enumerate(nums):
        while len(st) > 1 and nums[st[-1]] >= x:  # 改成 >= 就是求严格小于
            st.pop()
        left[i] = st[-1]
        st.append(i)
    
    right = [0] * n
    st = [n]
    for i in range(n - 1, -1, -1):
        x = nums[i]
        while len(st) > 1 and nums[st[-1]] >= x:
            st.pop()
        right[i] = st[-1]
        st.append(i)
    
    return left, right


# 示例使用
if __name__ == "__main__":
    nums = [2, 1, 5, 6, 2, 3]
    left, right = nearest_greater(nums)
    print(f"数组: {nums}")
    print(f"左侧更大元素下标: {left}")
    print(f"右侧更大元素下标: {right}")
