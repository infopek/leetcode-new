#include "./models/list_node.h"

#include <iostream>
#include <utility>
#include <vector>

// https://leetcode.com/problems/remove-duplicates-from-sorted-list/
ListNode* deleteDuplicates(ListNode* head)
{
    ListNode* curr = head;
    while (curr)
    {
        while (curr->next && curr->next->val == curr->val)
        {
            // Duplicate found -> unlink next node
            curr->next = curr->next->next;
        }

        curr = curr->next;
    }

    return head;
}

// https://leetcode.com/problems/merge-sorted-array/
void merge(std::vector<int>& nums1, int m, std::vector<int>& nums2, int n) 
{
    std::vector<int> merged(static_cast<size_t>(m + n));

    size_t i = 0;
    size_t j = 0;
    size_t mergedIdx = 0;
    while (i < m && j < n)
    {
        if (nums1[i] < nums2[j])
        {
            merged[mergedIdx] = nums1[i];
            ++mergedIdx;
            ++i;
        }
        else if (nums1[i] > nums2[j])
        {
            merged[mergedIdx] = nums2[j];
            ++mergedIdx;
            ++j;
        }
        else
        {
            merged[mergedIdx] = nums1[i];
            merged[mergedIdx + 1] = nums2[j];
            mergedIdx += 2;
            ++i;
            ++j;
        }
    }
    while (i < m)
    {
        // Fill with remaining elements from nums1
        merged[mergedIdx] = nums1[i];
        ++mergedIdx;
        ++i;
    }
    while (j < n)
    {
        // Fill with remaining elements from nums2
        merged[mergedIdx] = nums2[j];
        ++mergedIdx;
        ++j;
    }

    nums1 = std::move(merged);
}

int main()
{
    std::vector nums1{ 2, 0 };
    std::vector nums2{ 1 };

    merge(nums1, 1, nums2, 1);

}
