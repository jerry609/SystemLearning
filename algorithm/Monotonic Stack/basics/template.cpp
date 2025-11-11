#include <vector>
#include <stack>
#include <iostream>
using namespace std;

/**
 * 返回数组中每个元素左侧和右侧最近的严格大于它的元素的下标
 * 
 * 时间复杂度: O(n)
 * 空间复杂度: O(n)
 * 
 * @param nums 输入数组
 * @return pair<vector<int>, vector<int>>
 *         left[i] - nums[i] 左侧最近的严格大于 nums[i] 的数的下标，不存在则为 -1
 *         right[i] - nums[i] 右侧最近的严格大于 nums[i] 的数的下标，不存在则为 n
 */
pair<vector<int>, vector<int>> nearestGreater(const vector<int>& nums) {
    int n = nums.size();
    
    // left[i] 是 nums[i] 左侧最近的严格大于 nums[i] 的数的下标，若不存在则为 -1
    vector<int> left(n);
    stack<int> st;
    st.push(-1); // 哨兵
    for (int i = 0; i < n; i++) {
        int x = nums[i];
        while (st.size() > 1 && nums[st.top()] <= x) { // 如果求严格小于，改成 >=
            st.pop();
        }
        left[i] = st.top();
        st.push(i);
    }
    
    // right[i] 是 nums[i] 右侧最近的严格大于 nums[i] 的数的下标，若不存在则为 n
    vector<int> right(n);
    while (!st.empty()) st.pop(); // 清空栈
    st.push(n); // 哨兵
    for (int i = n - 1; i >= 0; i--) {
        int x = nums[i];
        while (st.size() > 1 && nums[st.top()] <= x) {
            st.pop();
        }
        right[i] = st.top();
        st.push(i);
    }
    
    return {left, right};
}

/**
 * 返回数组中每个元素左侧和右侧最近的严格小于它的元素的下标
 */
pair<vector<int>, vector<int>> nearestSmaller(const vector<int>& nums) {
    int n = nums.size();
    
    vector<int> left(n);
    stack<int> st;
    st.push(-1);
    for (int i = 0; i < n; i++) {
        int x = nums[i];
        while (st.size() > 1 && nums[st.top()] >= x) { // 改成 >= 就是求严格小于
            st.pop();
        }
        left[i] = st.top();
        st.push(i);
    }
    
    vector<int> right(n);
    while (!st.empty()) st.pop();
    st.push(n);
    for (int i = n - 1; i >= 0; i--) {
        int x = nums[i];
        while (st.size() > 1 && nums[st.top()] >= x) {
            st.pop();
        }
        right[i] = st.top();
        st.push(i);
    }
    
    return {left, right};
}

// 示例使用
int main() {
    vector<int> nums = {2, 1, 5, 6, 2, 3};
    auto [left, right] = nearestGreater(nums);
    
    cout << "数组: ";
    for (int x : nums) cout << x << " ";
    cout << "\n左侧更大元素下标: ";
    for (int x : left) cout << x << " ";
    cout << "\n右侧更大元素下标: ";
    for (int x : right) cout << x << " ";
    cout << endl;
    
    return 0;
}
