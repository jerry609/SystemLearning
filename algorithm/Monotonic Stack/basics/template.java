import java.util.*;

public class MonotonicStack {
    
    /**
     * 返回数组中每个元素左侧和右侧最近的严格大于它的元素的下标
     * 
     * 时间复杂度: O(n)
     * 空间复杂度: O(n)
     * 
     * @param nums 输入数组
     * @return int[][] {left, right}
     *         left[i] - nums[i] 左侧最近的严格大于 nums[i] 的数的下标，不存在则为 -1
     *         right[i] - nums[i] 右侧最近的严格大于 nums[i] 的数的下标，不存在则为 n
     */
    public static int[][] nearestGreater(int[] nums) {
        int n = nums.length;
        
        // left[i] 是 nums[i] 左侧最近的严格大于 nums[i] 的数的下标，若不存在则为 -1
        int[] left = new int[n];
        Deque<Integer> st = new ArrayDeque<>();
        st.push(-1); // 哨兵
        for (int i = 0; i < n; i++) {
            int x = nums[i];
            while (st.size() > 1 && nums[st.peek()] <= x) { // 如果求严格小于，改成 >=
                st.pop();
            }
            left[i] = st.peek();
            st.push(i);
        }
        
        // right[i] 是 nums[i] 右侧最近的严格大于 nums[i] 的数的下标，若不存在则为 n
        int[] right = new int[n];
        st.clear();
        st.push(n); // 哨兵
        for (int i = n - 1; i >= 0; i--) {
            int x = nums[i];
            while (st.size() > 1 && nums[st.peek()] <= x) {
                st.pop();
            }
            right[i] = st.peek();
            st.push(i);
        }
        
        return new int[][]{left, right};
    }
    
    /**
     * 返回数组中每个元素左侧和右侧最近的严格小于它的元素的下标
     */
    public static int[][] nearestSmaller(int[] nums) {
        int n = nums.length;
        
        int[] left = new int[n];
        Deque<Integer> st = new ArrayDeque<>();
        st.push(-1);
        for (int i = 0; i < n; i++) {
            int x = nums[i];
            while (st.size() > 1 && nums[st.peek()] >= x) { // 改成 >= 就是求严格小于
                st.pop();
            }
            left[i] = st.peek();
            st.push(i);
        }
        
        int[] right = new int[n];
        st.clear();
        st.push(n);
        for (int i = n - 1; i >= 0; i--) {
            int x = nums[i];
            while (st.size() > 1 && nums[st.peek()] >= x) {
                st.pop();
            }
            right[i] = st.peek();
            st.push(i);
        }
        
        return new int[][]{left, right};
    }
    
    // 示例使用
    public static void main(String[] args) {
        int[] nums = {2, 1, 5, 6, 2, 3};
        int[][] result = nearestGreater(nums);
        System.out.println("数组: " + Arrays.toString(nums));
        System.out.println("左侧更大元素下标: " + Arrays.toString(result[0]));
        System.out.println("右侧更大元素下标: " + Arrays.toString(result[1]));
    }
}
