#include "./models/list_node.h"
#include "./models/tree_node.h"

#include <algorithm>
#include <bit>
#include <bitset>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_map>
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

// https://leetcode.com/problems/linked-list-cycle/
namespace p141
{
	bool hasCycle(ListNode* head)
	{
		ListNode* slow = head;
		ListNode* fast = head;
		while (fast && fast->next)
		{
			slow = slow->next;
			fast = fast->next->next;

			if (fast && slow == fast)
				return true;
		}

		return false;
	}
}

// https://leetcode.com/problems/binary-tree-preorder-traversal/
namespace p144
{
	void dfs(TreeNode* node, std::vector<int>& traversal)
	{
		if (!node)
			return;

		traversal.push_back(node->val);
		dfs(node->left, traversal);
		dfs(node->right, traversal);
	}

	std::vector<int> preorderTraversal(TreeNode* root)
	{
		std::vector<int> traversal{};
		dfs(root, traversal);
		return traversal;
	}
}

// https://leetcode.com/problems/binary-tree-postorder-traversal/
namespace p145
{
	void dfs(TreeNode* node, std::vector<int>& traversal)
	{
		if (!node)
			return;

		dfs(node->left, traversal);
		dfs(node->right, traversal);
		traversal.push_back(node->val);
	}

	std::vector<int> postorderTraversal(TreeNode* root)
	{
		std::vector<int> traversal{};
		dfs(root, traversal);
		return traversal;
	}
}

// https://leetcode.com/problems/intersection-of-two-linked-lists/
namespace p160
{
	int calcLength(ListNode* head)
	{
		ListNode* ptr = head;
		int length = 0;
		while (ptr)
		{
			++length;
			ptr = ptr->next;
		}

		return length;
	}

	ListNode* getIntersectionNode(ListNode* headA, ListNode* headB)
	{
		int lengthA = calcLength(headA);
		int lengthB = calcLength(headB);
		int diff = std::abs(lengthA - lengthB);

		ListNode* currA = headA;
		ListNode* currB = headB;
		if (lengthA >= lengthB)
		{
			while (currA && diff > 0)
			{
				currA = currA->next;
				--diff;
			}
		}
		else
		{
			while (currB && diff > 0)
			{
				currB = currB->next;
				--diff;
			}
		}

		while (currA && currB)
		{
			if (currA == currB)
				return currA;
			currA = currA->next;
			currB = currB->next;
		}

		return nullptr;
	}
}

// https://leetcode.com/problems/excel-sheet-column-title/
namespace p168
{
	std::string convertToTitle(int columnNumber)
	{
		auto log = [](double x, double base) -> double
			{
				return std::log(x) / std::log(base);
			};

		const int alphabetLength = 26;
		double operand = (alphabetLength - 1) * static_cast<double>(columnNumber) + 1;
		int titleLength = static_cast<int>(std::floor(log(operand, alphabetLength)));
		std::string title{};
		title.resize(titleLength);

		int idx = titleLength - 1;
		while (idx >= 0)
		{
			--columnNumber;
			int div = columnNumber / alphabetLength;
			int rem = columnNumber % alphabetLength;
			title[idx] = 'A' + rem;

			columnNumber = div;
			--idx;
		}

		return title;
	}
}

// https://leetcode.com/problems/majority-element/
namespace p169
{
	int majorityElement(std::vector<int>& nums)
	{
		std::unordered_map<int, int> seen{};
		const int majorityCount = nums.size() / 2;
		for (const auto n : nums)
		{
			++seen[n];
			if (seen[n] > majorityCount)
				return n;
		}

		return -1;
	}

	int majorityElement2(std::vector<int>& nums)
	{
		// Boyer-Moore voting algorithm: https://en.wikipedia.org/wiki/Boyer%E2%80%93Moore_majority_vote_algorithm
		int candidate = nums[0];
		int count = 1;
		for (size_t i = 1; i < nums.size(); i++)
		{
			if (count == 0)
			{
				candidate = nums[i];
				++count;
			}
			else if (nums[i] == candidate)
			{
				++count;
			}
			else
			{
				--count;
			}
		}

		return candidate;
	}
}

// https://leetcode.com/problems/excel-sheet-column-number/
namespace p171
{
	int titleToNumber(const std::string& columnTitle)
	{
		const int alphabetLength = 26;

		int num = 0;
		int place = std::powl(alphabetLength, columnTitle.length() - 1);
		for (auto it = columnTitle.begin(); it != columnTitle.end(); ++it)
		{
			int n = (*it - 'A' + 1) * place;
			num += n;

			place /= alphabetLength;
		}

		return num;
	}
}

// https://leetcode.com/problems/reverse-bits/
namespace p190
{
	uint32_t reverseBits(uint32_t n)
	{
		unsigned s = 8 * sizeof(n);	// 8 bits in a byte
		s >>= 1;	// half will be needed for first iteration involving mask

		uint32_t mask = ~0;
		while (s > 0)
		{
			mask ^= mask << s;
			n = ((n >> s) & mask) | ((n << s) & ~mask);

			s >>= 1;
		}

		return n;
	}
}

// https://leetcode.com/problems/number-of-1-bits/
namespace p191
{
	int hammingWeight(int n)
	{
		std::bitset<32> bits(n);
		return bits.count();
	}

	int hammingWeight2(int n)
	{
		uint32_t mask = 1 << (8 * sizeof(n) - 1);
		int count = 0;
		while (mask > 0)
		{
			if ((n & mask) != 0)
				++count;

			mask >>= 1;
		}

		return count;
	}

	int hammingWeight3(int n)
	{
		int count = 0;
		while (n != 0)
		{
			++count;
			n &= (n - 1);	// flip least significant bit
		}
		return count;
	}
}

// https://leetcode.com/problems/happy-number/
namespace p202
{
	int squareSum(int n)
	{
		int sum = 0;
		while (n != 0)
		{
			int digit = n % 10;
			sum += digit * digit;
			n /= 10;
		}

		return sum;
	}

	bool isHappy(int n)
	{
		std::unordered_map<int, bool> seen{};
		while (n != 1)
		{
			int sum = squareSum(n);

			if (seen[sum])
				return false;

			seen[sum] = true;
			n = sum;
		}

		return true;
	}

	bool isHappy2(int n)
	{
		int slow = squareSum(n);
		int fast = squareSum(squareSum(n));
		while (slow != fast && fast != 1)
		{
			slow = squareSum(slow);
			fast = squareSum(squareSum(fast));
		}

		return fast == 1;
	}
}

// https://leetcode.com/problems/remove-linked-list-elements/
namespace p203
{
	ListNode* removeElements(ListNode* head, int val)
	{
		// Delete from front first
		while (head && head->val == val)
			head = head->next;

		// Process rest of the list
		ListNode* prev = head;
		ListNode* curr = head;
		while (curr)
		{
			while (curr && curr->val == val)
			{
				// Unlink
				prev->next = curr->next;
				curr = curr->next;
			}

			if (curr)
			{
				prev = curr;
				curr = curr->next;
			}
		}

		return head;
	}
}

// https://leetcode.com/problems/contains-duplicate/
namespace p217
{
	bool containsDuplicate(std::vector<int>& nums)
	{
		std::unordered_map<int, int> count{};
		for (const auto n : nums)
		{
			++count[n];
			if (count[n] >= 2)
				return true;
		}

		return false;
	}

	bool containsDuplicate2(std::vector<int>& nums)
	{
		std::unordered_map<int, int> count{};
		for (const auto n : nums)
		{
			++count[n];
			if (count[n] >= 2)
				return true;
		}

		return false;
	}
}

int main()
{
	for (int i = 1; i < 1000; i++)
	{
		std::cout << i << ": " << p202::isHappy2(i) << '\n';
	}
}
