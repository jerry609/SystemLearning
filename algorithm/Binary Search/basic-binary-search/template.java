// Java 二分查找模板

import java.util.Arrays;

public class BinarySearchTemplate {
    
    /**
     * 返回 >= target 的第一个元素的下标
     * 如果不存在，返回 nums.length
     */
    public static int lowerBound(int[] nums, int target) {
        int left = 0, right = nums.length;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) { // 红色区域
                left = mid + 1;
            } else { // 蓝色区域，nums[mid] >= target
                right = mid;
            }
        }
        return left;
    }
    
    /**
     * 闭区间写法
     */
    public static int lowerBoundClosed(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return left;
    }
    
    /**
     * 返回 >= target 的第一个元素的下标
     */
    public static int firstGreaterOrEqual(int[] nums, int target) {
        return lowerBound(nums, target);
    }
    
    /**
     * 返回 > target 的第一个元素的下标
     */
    public static int firstGreater(int[] nums, int target) {
        return lowerBound(nums, target + 1);
    }
    
    /**
     * 返回 < target 的最后一个元素的下标
     */
    public static int lastLess(int[] nums, int target) {
        return lowerBound(nums, target) - 1;
    }
    
    /**
     * 返回 <= target 的最后一个元素的下标
     */
    public static int lastLessOrEqual(int[] nums, int target) {
        return lowerBound(nums, target + 1) - 1;
    }
    
    /**
     * 统计 < target 的元素个数
     */
    public static int countLess(int[] nums, int target) {
        return lowerBound(nums, target);
    }
    
    /**
     * 统计 <= target 的元素个数
     */
    public static int countLessOrEqual(int[] nums, int target) {
        return lowerBound(nums, target + 1);
    }
    
    /**
     * 统计 >= target 的元素个数
     */
    public static int countGreaterOrEqual(int[] nums, int target) {
        return nums.length - lowerBound(nums, target);
    }
    
    /**
     * 统计 > target 的元素个数
     */
    public static int countGreater(int[] nums, int target) {
        return nums.length - lowerBound(nums, target + 1);
    }
    
    public static void main(String[] args) {
        int[] nums = {1, 3, 3, 5, 7, 7, 7, 9};
        int target = 7;
        
        System.out.println("数组: " + Arrays.toString(nums));
        System.out.println("目标值: " + target + "\n");
        
        // 查找位置
        System.out.println(">= " + target + " 的第一个位置: " + firstGreaterOrEqual(nums, target));
        System.out.println("> " + target + " 的第一个位置: " + firstGreater(nums, target));
        System.out.println("< " + target + " 的最后一个位置: " + lastLess(nums, target));
        System.out.println("<= " + target + " 的最后一个位置: " + lastLessOrEqual(nums, target));
        
        System.out.println();
        
        // 统计个数
        System.out.println("< " + target + " 的元素个数: " + countLess(nums, target));
        System.out.println("<= " + target + " 的元素个数: " + countLessOrEqual(nums, target));
        System.out.println(">= " + target + " 的元素个数: " + countGreaterOrEqual(nums, target));
        System.out.println("> " + target + " 的元素个数: " + countGreater(nums, target));
        
        // 使用标准库
        System.out.println("\n使用 Arrays.binarySearch: " + Arrays.binarySearch(nums, target));
    }
}
