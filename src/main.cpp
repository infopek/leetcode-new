#include "./models/list_node.h"
#include "./models/tree_node.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

// https://leetcode.com/problems/remove-duplicates-from-sorted-list/
namespace p83
{
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
}

// https://leetcode.com/problems/merge-sorted-array/
namespace p88
{
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
}

// https://leetcode.com/problems/same-tree/
namespace p100
{
	void dfs(TreeNode* node, std::string& repr)
	{
		if (!node)
		{
			repr += 'N';
			return;
		}

		repr += std::to_string(node->val);
		dfs(node->left, repr);
		dfs(node->right, repr);
	}

	bool isSameTree(TreeNode* p, TreeNode* q)
	{
		std::string pString{};
		std::string qString{};
		dfs(p, pString);
		dfs(q, qString);

		return pString == qString;
	}

	// Other solution
	bool isSameTree2(TreeNode* p, TreeNode* q)
	{
		if (!p && !q)
			return true;

		if (!p || !q || p->val != q->val)
			return false;

		return isSameTree2(p->left, q->left) && isSameTree2(p->right, q->right);
	}
}

// https://leetcode.com/problems/symmetric-tree/
namespace p101
{
	bool isSymmetricDFS(TreeNode* p, TreeNode* q)
	{
		if (!p && !q)
			return true;

		if (!p || !q || p->val != q->val)
			return false;

		return isSymmetricDFS(p->left, q->right) && isSymmetricDFS(p->right, q->left);
	}

	bool isSymmetric(TreeNode* root)
	{
		return isSymmetricDFS(root->left, root->right);
	}
}

// https://leetcode.com/problems/symmetric-tree/
namespace p104
{
	int maxDepth(TreeNode* root)
	{
		if (!root)
			return 0;

		int leftHeight = 1 + maxDepth(root->left);
		int rightHeight = 1 + maxDepth(root->right);
		return std::max(leftHeight, rightHeight);
	}
}

// https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/
namespace p108
{
	TreeNode* buildTree(const std::vector<int>& nums, int left, int right)
	{
		if (left > right)
			return nullptr;

		int mid = (right - left) / 2 + left;

		TreeNode* root = new TreeNode(nums[mid]);
		root->left = buildTree(nums, left, mid - 1);
		root->right = buildTree(nums, mid + 1, right);

		return root;
	}

	TreeNode* sortedArrayToBST(const std::vector<int>& nums)
	{
		TreeNode* root = buildTree(nums, 0, nums.size() - 1);
		return root;
	}
}

// https://leetcode.com/problems/balanced-binary-tree/
namespace p110
{
	int getDepth(TreeNode* root, bool& isBalanced)
	{
		if (!root || !isBalanced)
			return 0;

		int leftHeight = 1 + getDepth(root->left, isBalanced);
		int rightHeight = 1 + getDepth(root->right, isBalanced);
		if (std::abs(leftHeight - rightHeight) > 1)
			isBalanced = false;

		return std::max(leftHeight, rightHeight);
	}

	bool isBalanced(TreeNode* root)
	{
		bool isBalanced = true;
		int _ = getDepth(root, isBalanced);
		return isBalanced;
	}
}

// https://leetcode.com/problems/minimum-depth-of-binary-tree/
namespace p111
{
	void minDepthDFS(TreeNode* node, int& minDepth, int currDepth)
	{
		if (!node)
			return;
		if (!node->left && !node->right)
			minDepth = std::min(minDepth, currDepth);

		minDepthDFS(node->left, minDepth, currDepth + 1);
		minDepthDFS(node->right, minDepth, currDepth + 1);
	}

	int minDepth(TreeNode* root)
	{
		if (!root)
			return 0;

		int minDepth = INT_MAX;
		minDepthDFS(root, minDepth, 1);
		return minDepth;
	}
}

// https://leetcode.com/problems/pascals-triangle/
namespace p118
{
	std::vector<std::vector<int>> generate(int numRows)
	{
		std::vector<std::vector<int>> triangle(numRows);
		for (size_t i = 0; i < static_cast<size_t>(numRows); i++)
		{
			triangle[i].resize(i + 1);

			// First and last entry in a row is always 1
			triangle[i][0] = 1;
			triangle[i][i] = 1;
			for (size_t j = 1; j < i; j++)
			{
				triangle[i][j] = triangle[i - 1][j - 1] + triangle[i - 1][j];
			}
		}

		return triangle;
	}
}

// https://leetcode.com/problems/pascals-triangle-ii/
namespace p119
{
	std::vector<int> getRow(int rowIndex)
	{
		// Helper lambda to calculate n choose k
		auto nCk = [](size_t n, size_t k) -> size_t {
			if (k > n - k)
				k = n - k;	// symmetry

			size_t result = 1;
			for (size_t i = 0; i < k; i++)
			{
				result *= (n - i);
				result /= (i + 1);
			}

			return result;
			};

		size_t rowSize = static_cast<size_t>(rowIndex + 1);
		std::vector<int> row(rowSize);

		size_t mid = (rowSize % 2 == 0)
			? rowSize / 2
			: rowSize / 2 + 1;
		for (size_t i = 0; i < mid; i++)
		{
			row[i] = nCk(rowSize - 1, i);
			row[rowSize - i - 1] = row[i];
		}

		return row;
	}
}

// https://leetcode.com/problems/valid-palindrome/
namespace p125
{
	bool isPalindrome(const std::string& phrase)
	{
		int left = 0;
		int right = phrase.size() - 1;
		while (left < right)
		{
			while (left < right && !std::isalnum(phrase[left]))
				++left;

			while (left < right && !std::isalnum(phrase[right]))
				--right;

			if (left < right && std::tolower(phrase[left]) != std::tolower(phrase[right]))
				return false;

			++left;
			--right;
		}

		return true;
	}
}

// https://leetcode.com/problems/single-number/
namespace p136
{
	int singleNumber(std::vector<int>& nums)
	{
		int sum = 0;
		for (const auto n : nums)
			sum ^= n;

		return sum;
	}
}

int main()
{
	std::vector nums{ 4, 1, 2, 1, 2 };
	int res = p136::singleNumber(nums);

}
