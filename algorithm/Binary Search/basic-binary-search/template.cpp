// C++ 二分查找模板

#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

class BinarySearchTemplate {
public:
    /**
     * 返回 >= target 的第一个元素的下标
     * 如果不存在，返回 nums.size()
     */
    static int lowerBound(const vector<int>& nums, int target) {
        int left = 0, right = nums.size();
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
    static int lowerBoundClosed(const vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;
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
    static int firstGreaterOrEqual(const vector<int>& nums, int target) {
        return lowerBound(nums, target);
    }
    
    /**
     * 返回 > target 的第一个元素的下标
     */
    static int firstGreater(const vector<int>& nums, int target) {
        return lowerBound(nums, target + 1);
    }
    
    /**
     * 返回 < target 的最后一个元素的下标
     */
    static int lastLess(const vector<int>& nums, int target) {
        return lowerBound(nums, target) - 1;
    }
    
    /**
     * 返回 <= target 的最后一个元素的下标
     */
    static int lastLessOrEqual(const vector<int>& nums, int target) {
        return lowerBound(nums, target + 1) - 1;
    }
    
    /**
     * 统计 < target 的元素个数
     */
    static int countLess(const vector<int>& nums, int target) {
        return lowerBound(nums, target);
    }
    
    /**
     * 统计 <= target 的元素个数
     */
    static int countLessOrEqual(const vector<int>& nums, int target) {
        return lowerBound(nums, target + 1);
    }
    
    /**
     * 统计 >= target 的元素个数
     */
    static int countGreaterOrEqual(const vector<int>& nums, int target) {
        return nums.size() - lowerBound(nums, target);
    }
    
    /**
     * 统计 > target 的元素个数
     */
    static int countGreater(const vector<int>& nums, int target) {
        return nums.size() - lowerBound(nums, target + 1);
    }
};

int main() {
    vector<int> nums = {1, 3, 3, 5, 7, 7, 7, 9};
    int target = 7;
    
    cout << "数组: ";
    for (int x : nums) cout << x << " ";
    cout << "\n目标值: " << target << "\n\n";
    
    // 查找位置
    cout << ">= " << target << " 的第一个位置: " 
         << BinarySearchTemplate::firstGreaterOrEqual(nums, target) << "\n";
    cout << "> " << target << " 的第一个位置: " 
         << BinarySearchTemplate::firstGreater(nums, target) << "\n";
    cout << "< " << target << " 的最后一个位置: " 
         << BinarySearchTemplate::lastLess(nums, target) << "\n";
    cout << "<= " << target << " 的最后一个位置: " 
         << BinarySearchTemplate::lastLessOrEqual(nums, target) << "\n";
    
    cout << "\n";
    
    // 统计个数
    cout << "< " << target << " 的元素个数: " 
         << BinarySearchTemplate::countLess(nums, target) << "\n";
    cout << "<= " << target << " 的元素个数: " 
         << BinarySearchTemplate::countLessOrEqual(nums, target) << "\n";
    cout << ">= " << target << " 的元素个数: " 
         << BinarySearchTemplate::countGreaterOrEqual(nums, target) << "\n";
    cout << "> " << target << " 的元素个数: " 
         << BinarySearchTemplate::countGreater(nums, target) << "\n";
    
    // 使用标准库
    cout << "\n使用 lower_bound: " 
         << lower_bound(nums.begin(), nums.end(), target) - nums.begin() << "\n";
    cout << "使用 upper_bound: " 
         << upper_bound(nums.begin(), nums.end(), target) - nums.begin() << "\n";
    
    return 0;
}
