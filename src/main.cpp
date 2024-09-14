#include "./models/list_node.h"
#include "./models/tree_node.h"
#include "./models/node.h"

#include <algorithm>
#include <bit>
#include <bitset>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <queue>
#include <ranges>
#include <stack>
#include <stdlib.h>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
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
}

// https://leetcode.com/problems/count-complete-tree-nodes/
namespace p222
{
	void dfs(TreeNode* node, int& count)
	{
		if (!node)
			return;

		++count;
		dfs(node->left, count);
		dfs(node->right, count);
	}

	int countNodes(TreeNode* root)
	{
		int count = 0;
		dfs(root, count);
		return count;
	}

	int countNodes2(TreeNode* root)
	{
		if (!root)
			return 0;

		int leftCount = countNodes(root->left);
		int rightCount = countNodes(root->right);
		return leftCount + rightCount + 1;
	}

	int countNodes3(TreeNode* root)
	{
		if (!root)
			return 0;

		TreeNode* left = root;
		TreeNode* right = root;
		int leftHeight = 0;
		int rightHeight = 0;
		while (left)
		{
			++leftHeight;
			left = left->left;
		}
		while (right)
		{
			++rightHeight;
			right = right->right;
		}

		if (leftHeight == rightHeight)
			return (1 << leftHeight) - 1;	// 2^h - 1

		return 1 + countNodes3(root->left) + countNodes3(root->right);
	}
}

// https://leetcode.com/problems/implement-stack-using-queues/
namespace p225
{
	class MyStack1
	{
	public:
		MyStack1()
		{

		}

		void push(int x)
		{
			while (!m_queue1.empty())
			{
				m_queue2.push(m_queue1.front());
				m_queue1.pop();
			}
			m_queue1.push(x);
			while (!m_queue2.empty())
			{
				m_queue1.push(m_queue2.front());
				m_queue2.pop();
			}
		}

		int pop()
		{
			int top = m_queue1.front();
			m_queue1.pop();
			return top;
		}

		int top()
		{
			return m_queue1.front();
		}

		bool empty()
		{
			return m_queue1.empty();
		}

	private:
		std::queue<int> m_queue1{};
		std::queue<int> m_queue2{};
	};

	class MyStack2
	{
	public:
		MyStack2()
		{

		}

		void push(int x)
		{
			m_queue.push(x);
			for (size_t i = 0; i < m_queue.size() - 1; i++)
			{
				m_queue.push(m_queue.front());
				m_queue.pop();
			}
		}

		int pop()
		{
			int top = m_queue.front();
			m_queue.pop();
			return top;
		}

		int top()
		{
			return m_queue.front();
		}

		bool empty()
		{
			return m_queue.empty();
		}

	private:
		std::queue<int> m_queue{};
	};
}

// https://leetcode.com/problems/invert-binary-tree/
namespace p226
{
	void invertDFS(TreeNode* node)
	{
		if (!node)
			return;

		invertDFS(node->left);
		invertDFS(node->right);
		std::swap(node->left, node->right);
	}

	TreeNode* invertTree(TreeNode* root)
	{
		invertDFS(root);
		return root;
	}
}

// https://leetcode.com/problems/summary-ranges/description/
namespace p228
{
	std::vector<std::string> summaryRanges(std::vector<int>& nums)
	{
		const size_t size = nums.size();

		std::vector<std::string> ranges{};
		size_t left = 0;
		size_t right = 0;
		while (right < size)
		{
			while (right < size - 1 && nums[right] + 1 == nums[right + 1])
				++right;

			std::string range = (left != right)
				? std::to_string(nums[left]) + "->" + std::to_string(nums[right])
				: std::to_string(nums[left]);
			ranges.push_back(range);

			++right;
			left = right;
		}

		return ranges;
	}
}

// https://leetcode.com/problems/power-of-two/
namespace p231
{
	bool isPowerOfTwo(int n)
	{
		if (n <= 0)
			return false;

		int count = 0;
		while (n > 0 && count < 2)
		{
			count += (n & 1);
			n >>= 1;
		}

		return count == 1;
	}

	bool isPowerOfTwo2(int n)
	{
		return (n > 0) && ((n - 1) & n) == 0;
	}
}

namespace p232
{
	class MyQueue
	{
	public:
		MyQueue()
		{

		}

		void push(int x)
		{
			while (!m_stack1.empty())
			{
				m_stack2.push(m_stack1.top());
				m_stack1.pop();
			}
			m_stack1.push(x);
			while (!m_stack2.empty())
			{
				m_stack1.push(m_stack2.top());
				m_stack2.pop();
			}
		}

		int pop()
		{
			int front = m_stack1.top();
			m_stack1.pop();
			return front;
		}

		int peek()
		{
			return m_stack1.top();
		}

		bool empty()
		{
			return m_stack1.empty();
		}

	private:
		std::stack<int> m_stack1{};
		std::stack<int> m_stack2{};
	};

	class MyQueue2
	{
	public:
		MyQueue2()
		{

		}

		void push(int x)
		{
			m_stack1.push(x);
		}

		int pop()
		{
			if (m_stack2.empty())
			{
				while (!m_stack1.empty())
				{
					m_stack2.push(m_stack1.top());
					m_stack1.pop();
				}
			}

			int front = m_stack2.top();
			m_stack2.pop();
			return front;
		}

		int peek()
		{
			if (m_stack2.empty())
			{
				while (!m_stack1.empty())
				{
					m_stack2.push(m_stack1.top());
					m_stack1.pop();
				}
			}

			return m_stack2.top();
		}

		bool empty()
		{
			return m_stack1.empty() && m_stack2.empty();
		}

	private:
		std::stack<int> m_stack1{};
		std::stack<int> m_stack2{};
	};
}

// https://leetcode.com/problems/valid-anagram/
namespace p242
{
	bool isAnagram(std::string& s, std::string& t)
	{
		std::sort(s.begin(), s.end());
		std::sort(t.begin(), t.end());
		return s == t;
	}

	std::vector<int> getLetterCount(const std::string& str)
	{
		const size_t alphabetSize = 26;
		std::vector<int> count(alphabetSize);
		for (const auto c : str)
			++count[c - static_cast<size_t>('a')];

		return count;
	}

	bool isAnagram2(const std::string& s, const std::string& t)
	{
		std::vector<int> sCount = getLetterCount(s);
		std::vector<int> tCount = getLetterCount(t);
		return sCount == tCount;
	}
}

// https://leetcode.com/problems/binary-tree-paths/
namespace p257
{
	void collectPathsDFS(TreeNode* node, std::vector<std::string>& paths, const std::string& currPath = "")
	{
		if (!node)
			return;

		std::string path{ currPath + std::to_string(node->val) };
		if (!node->left && !node->right)
			paths.push_back(path);

		path += "->";
		collectPathsDFS(node->left, paths, path);
		collectPathsDFS(node->right, paths, path);
	}

	std::vector<std::string> binaryTreePaths(TreeNode* root)
	{
		std::vector<std::string> paths{};
		collectPathsDFS(root, paths);
		return paths;
	}
}

// https://leetcode.com/problems/add-digits/
namespace p258
{
	int addDigits(int num)
	{
		while (num > 9)
		{
			int sum = 0;
			while (num != 0)
			{
				sum += num % 10;
				num /= 10;
			}

			num = sum;
		}

		return num;
	}

	int addDigits2(int num)
	{
		if (num == 0)
			return 0;
		else if (num % 9 == 0)
			return 9;

		return num % 9;
	}
}

// https://leetcode.com/problems/ugly-number/
namespace p263
{
	bool isUgly(int n)
	{
		if (n <= 0)
			return false;

		int factors[]{ 2, 3, 5 };
		for (const auto factor : factors)
		{
			while (n != 1 && n % factor == 0)
				n /= factor;
		}

		return n == 1;
	}
}

// https://leetcode.com/problems/missing-number/
namespace p268
{
	// Solution 1: Sum of first n natural numbers formula
	int missingNumber(const std::vector<int>& nums)
	{
		const int size = nums.size();
		int sum = size * (size + 1) / 2;
		for (const auto n : nums)
			sum -= n;

		return sum;
	}
}

// https://leetcode.com/problems/move-zeroes
namespace p283
{
	// Solution 1: left and right pointers
	void moveZeroes(std::vector<int>& nums)
	{
		const int size = static_cast<int>(nums.size());

		int left = 0;
		while (left < size)
		{
			while (left < size && nums[left] != 0)
				++left;

			int right = left + 1;
			while (right < size && nums[right] == 0)
				++right;

			if (right < size)
				std::swap(nums[left], nums[right]);

			++left;
		}
	}

	// Solution 2: more efficient two pointer
	void moveZeroes2(std::vector<int>& nums)
	{
		int left = 0;
		for (int right = 0; right < nums.size(); right++)
		{
			if (nums[right] != 0)
			{
				std::swap(nums[left], nums[right]);
				++left;
			}
		}
	}
}

// https://leetcode.com/problems/word-pattern/
namespace p290
{
	std::vector<std::string> split(const std::string& str, const std::string& delim)
	{
		std::vector<std::string> result;
		size_t start = 0;

		for (size_t found = str.find(delim); found != std::string::npos; found = str.find(delim, start))
		{
			result.emplace_back(str.begin() + start, str.begin() + found);
			start = found + delim.size();
		}
		if (start != str.size())
			result.emplace_back(str.begin() + start, str.end());
		return result;
	}

	bool wordPattern(const std::string& pattern, const std::string& s)
	{
		std::unordered_map<char, std::string> charToString{};
		std::unordered_map<std::string, char> stringToChar{};
		std::vector<std::string> words = split(s, " ");

		const size_t length = s.length();
		size_t i = 0;
		for (const auto c : pattern)
		{
			std::string word = (i < words.size())
				? words[i]
				: "";
			auto it1 = charToString.find(c);
			auto it2 = stringToChar.find(word);
			if (it1 != charToString.end() && it2 != stringToChar.end())
			{
				if (it1->second != word || it2->second != c)
					return false;
			}
			else if (it1 == charToString.end() && it2 == stringToChar.end())
			{
				charToString[c] = word;
				stringToChar[word] = c;
			}
			else if (it1 != charToString.end() || it2 != stringToChar.end())
			{
				return false;
			}

			++i;
		}

		return i == words.size();
	}
}

// https://leetcode.com/problems/nim-game/
namespace p292
{
	bool canWinNim(int n)
	{
		return n % 4 != 0;
	}
}

// https://leetcode.com/problems/range-sum-query-immutable/
namespace p303
{
	// Solution 1: naive / iterative
	class NumArray
	{
	public:
		NumArray(const std::vector<int>& nums)
			: m_nums{ nums }
		{
		}

		int sumRange(int left, int right)
		{
			int sum = 0;
			for (int i = left; i <= right; i++)
				sum += m_nums[i];

			return sum;
		}

	private:
		std::vector<int> m_nums{};
	};

	// Solution 1: caching (TLE)
	class NumArray2
	{
	public:
		NumArray2(const std::vector<int>& nums)
			: m_nums{ nums }
		{
		}

		int sumRange(int left, int right)
		{
			if (auto it = m_cache.find(std::to_string(left) + std::to_string(right)); it != m_cache.end())
			{
				return m_cache[std::to_string(left) + std::to_string(right)];
			}
			else if (right - left == 1)
			{
				m_cache[std::to_string(left) + std::to_string(right)] = m_nums[left] + m_nums[right];
				return m_nums[left] + m_nums[right];
			}
			else if (left == right)
			{
				return m_nums[left];
			}

			int mid = left + (right - left) / 2;
			return sumRange(left, mid - 1) + sumRange(mid, right);
		}

	private:
		std::vector<int> m_nums{};
		std::unordered_map<std::string, int> m_cache{};
	};

	// Solution 3: prefix sum
	class NumArray3
	{
	public:
		NumArray3(const std::vector<int>& nums)
		{
			m_nums.resize(nums.size() + 1);
			m_nums[0] = 0;
			for (size_t i = 1; i < nums.size() + 1; i++)
				m_nums[i] = m_nums[i - 1] + nums[i - 1];
		}

		int sumRange(int left, int right)
		{
			return  m_nums[right] - m_nums[left - 1];
		}

	private:
		std::vector<int> m_nums{};
	};
}

// https://leetcode.com/problems/power-of-three/
namespace p326
{
	// Solution 1: divide until 1
	bool isPowerOfThree(int n)
	{
		if (n <= 0)
			return false;

		while (n % 3 == 0)
			n /= 3;

		return n == 1;
	}

	// Solution 2: largest power of 3 in 32-bit int
	bool isPowerOfThree2(int n)
	{
		const int largestPowThree = 1162261467;
		return n >= 0 && largestPowThree % n == 0;
	}
}

// https://leetcode.com/problems/counting-bits/
namespace p338
{
	// Solution 1: pop least significant bit
	std::vector<int> countBits(int n)
	{
		std::vector<int> bits(n + 1);
		for (int i = 0; i <= n; i++)
		{
			int count = 0;
			int j = i;
			while (j != 0)
			{
				j &= j - 1;
				++count;
			}

			bits[i] = count;
		}

		return bits;
	}

	// Solution 2: checking shifted bit count (pseudo recursion)
	std::vector<int> countBits2(int n)
	{
		std::vector<int> bits(n + 1);
		bits[0] = 0;
		for (int i = 1; i <= n; i++)
			bits[i] = bits[i / 2] + i % 2;

		return bits;
	}
}

// https://leetcode.com/problems/power-of-four/
namespace p342
{
	bool isPowerOfFour(int n)
	{
		return n > 0
			&& (n & 0xaaaaaaaa) == 0
			&& (n & (n - 1)) == 0;
	}
}

// https://leetcode.com/problems/reverse-string/
namespace p344
{
	void reverseString(std::vector<char>& s)
	{
		int left = 0;
		int right = s.size() - 1;
		while (left < right)
		{
			std::swap(s[left], s[right]);
			++left;
			--right;
		}
	}
}

// https://leetcode.com/problems/reverse-vowels-of-a-string/
namespace p345
{
	bool isVowel(char c)
	{
		c = std::tolower(c);
		return c == 'a'
			|| c == 'e'
			|| c == 'i'
			|| c == 'o'
			|| c == 'u';
	}

	std::string reverseVowels(const std::string& s)
	{
		std::string result = s;
		int left = 0;
		int right = s.size() - 1;
		while (left < right)
		{
			while (left < right && !isVowel(result[left]))
				++left;

			while (left < right && !isVowel(result[right]))
				--right;

			std::swap(result[left], result[right]);
			++left;
			--right;
		}

		return result;
	}
}

// https://leetcode.com/problems/intersection-of-two-arrays/
namespace p349
{
	// Solution 1: sorting, then bookish algo
	std::vector<int> intersection(std::vector<int>& nums1, std::vector<int>& nums2)
	{
		std::sort(nums1.begin(), nums1.end());
		std::sort(nums2.begin(), nums2.end());

		std::unordered_map<int, bool> seen{};
		std::vector<int> intersection{};
		intersection.reserve(std::min(nums1.size(), nums2.size()));
		int i = 0;
		int j = 0;
		while (i < nums1.size() && j < nums2.size())
		{
			if (nums1[i] < nums2[j])
			{
				++i;
			}
			else if (nums1[i] > nums2[j])
			{
				++j;
			}
			else
			{
				if (!seen[nums1[i]])
				{
					intersection.push_back(nums1[i]);
					seen[nums1[i]] = true;
				}

				++i;
				++j;
			}
		}

		return intersection;
	}

	// Solution 2: unordered_set
	std::vector<int> intersection2(const std::vector<int>& nums1, const std::vector<int>& nums2)
	{
		std::unordered_set<int> numsSet1(nums1.begin(), nums1.end());
		std::unordered_set<int> intersectionSet{};
		intersectionSet.reserve(std::min(nums1.size(), nums2.size()));
		for (const int n : nums2)
		{
			if (numsSet1.count(n) != 0)
				intersectionSet.insert(n);
		}

		return std::vector<int>(intersectionSet.begin(), intersectionSet.end());
	}
}

// https://leetcode.com/problems/intersection-of-two-arrays-ii/
namespace p350
{
	std::vector<int> intersect(std::vector<int>& nums1, std::vector<int>& nums2)
	{
		std::sort(nums1.begin(), nums1.end());
		std::sort(nums2.begin(), nums2.end());

		std::vector<int> intersection{};
		intersection.reserve(std::min(nums1.size(), nums2.size()));
		int i = 0;
		int j = 0;
		while (i < nums1.size() && j < nums2.size())
		{
			if (nums1[i] < nums2[j])
			{
				++i;
			}
			else if (nums1[i] > nums2[j])
			{
				++j;
			}
			else
			{
				intersection.push_back(nums1[i]);

				++i;
				++j;
			}
		}

		return intersection;
	}
}

// https://leetcode.com/problems/valid-perfect-square/
namespace p367
{
	// Solution 1: brute force
	bool isPerfectSquare(int num)
	{
		double start = (num % 2 == 0) ? 2 : 1;
		for (double i = start; i <= num / i; i += 2)
		{
			if (i == num / i)
				return true;
		}

		return false;
	}

	// Solution 2: binary search
	bool isPerfectSquare2(int num)
	{
		int left = 0;
		int right = num;
		while (left <= right)
		{
			int mid = left + (right - left) / 2;
			long square = static_cast<long>(mid) * mid;
			if (square < num)
				left = mid + 1;
			else if (square > num)
				right = mid - 1;
			else
				return true;
		}

		return false;
	}
}

// https://leetcode.com/problems/first-unique-character-in-a-string/
namespace p387
{
	// Solution 1: two hashmaps
	int firstUniqChar(const std::string& s)
	{
		std::unordered_map<char, int> count{};
		std::unordered_map<char, int> indices{};
		for (int i = 0; i < s.size(); ++i)
		{
			char c = s[i];
			if (auto it = indices.find(c); it == indices.end())
				indices[c] = i;

			++count[c];
		}

		for (const auto c : s)
		{
			if (count[c] == 1)
				return indices[c];
		}

		return -1;
	}

	// solution 2: queue
	int firstUniqChar2(const std::string& s)
	{
		std::unordered_map<char, int> freq{};
		std::queue<int> indices{};
		for (int i = 0; i < s.size(); ++i)
		{
			++freq[s[i]];
			if (freq[s[i]] == 1)
				indices.push(i);
		}

		while (!indices.empty())
		{
			int i = indices.front();
			indices.pop();
			if (freq[s[i]] == 1)
				return i;
		}

		return -1;
	}

	// Solution 3: one hashmap
	int firstUniqChar3(const std::string& s)
	{
		std::unordered_map<char, int> freq{};
		for (const auto c : s)
			++freq[c];

		for (size_t i = 0; i < s.size(); i++)
		{
			if (freq[s[i]] == 1)
				return i;
		}

		return -1;
	}

	// Solution 4: array
	int firstUniqChar4(const std::string& s)
	{
		const size_t alphabetSize = 26;
		int freq[alphabetSize]{};
		for (const auto c : s)
			++freq[c - 'a'];

		for (size_t i = 0; i < s.size(); i++)
		{
			if (freq[s[i] - 'a'] == 1)
				return i;
		}

		return -1;
	}
}

// https://leetcode.com/problems/find-the-difference/
namespace p389
{
	// Solution 1: two hashmaps
	char findTheDifference(const std::string& s, const std::string& t)
	{
		std::unordered_map<char, int> sLetters{};
		std::unordered_map<char, int> tLetters{};
		for (const auto c : s)
			++sLetters[c];
		for (const auto c : t)
			++tLetters[c];

		for (const auto c : t)
		{
			if (sLetters[c] != tLetters[c])
				return c;
		}

		return '0';
	}

	// Solution 2: one hashmap
	char findTheDifference2(const std::string& s, const std::string& t)
	{
		std::unordered_map<char, int> letters{};
		for (const auto c : s)
			++letters[c];
		for (const auto c : t)
			--letters[c];

		for (const auto c : t)
		{
			if (letters[c] != 0)
				return c;
		}

		return '0';
	}

	// Solution 3: array
	char findTheDifference3(const std::string& s, const std::string& t)
	{
		const size_t alphabetSize = 26;
		int freq[alphabetSize]{};
		for (const auto c : s)
			++freq[c - 'a'];
		for (const auto c : t)
			--freq[c - 'a'];

		for (const auto c : t)
		{
			if (freq[c - 'a'] != 0)
				return c;
		}

		return '0';
	}

	// Solution 4: diff propagation
	char findTheDifference4(const std::string& s, std::string& t)
	{
		for (size_t i = 0; i < s.size(); i++)
			t[i + 1] += t[i] - s[i];

		return t[t.size() - 1];
	}

	// Solution 5: xor
	char findTheDifference5(const std::string& s, const std::string& t)
	{
		char c = static_cast<char>(0);
		for (size_t i = 0; i < s.size(); i++)
		{
			c ^= s[i];
			c ^= t[i];
		}
		c ^= t[t.size() - 1];

		return c;
	}
}

// https://leetcode.com/problems/binary-watch/
namespace p401
{
	int countBits(int n)
	{
		int count = 0;
		while (n != 0)
		{
			++count;
			n &= (n - 1);
		}

		return count;
	}

	// Solution 1: bit counting
	std::vector<std::string> readBinaryWatch(int turnedOn)
	{
		std::vector<std::string> times{};
		for (int h = 0; h < 12; h++)
		{
			int hourBits = countBits(h);
			if (hourBits <= turnedOn)
			{
				int remaining = turnedOn - hourBits;
				for (int m = 0; m < 60; m++)
				{
					if (countBits(m) == remaining)
					{
						std::string time = std::to_string(h)
							+ ((m < 10) ? ":0" : ":")
							+ std::to_string(m);
						times.push_back(time);
					}
				}
			}
		}

		return times;
	}

	void backtrack(int idx, int hour, int minute, int remainingBits,
		const std::vector<int>& hours, const std::vector<int>& minutes, std::vector<std::string>& times)
	{
		if (remainingBits == 0)
		{
			std::string time = std::to_string(hour)
				+ ((minute < 10) ? ":0" : ":")
				+ std::to_string(minute);
			times.push_back(time);
		}

		for (int i = idx; i < hours.size() + minutes.size(); i++)
		{
			if (i < hours.size())
			{
				// Backtrack on hours
				hour += hours[i];
				if (hour < 12)
					backtrack(i + 1, hour, minute, remainingBits - 1, hours, minutes, times);
				hour -= hours[i];	// backtrack step
			}
			else
			{
				int minuteIdx = i - 4;
				minute += minutes[minuteIdx];
				if (minute < 60)
					backtrack(i + 1, hour, minute, remainingBits - 1, hours, minutes, times);
				minute -= minutes[minuteIdx];
			}
		}
	}

	// Solution 2: backtrack
	std::vector<std::string> readBinaryWatch2(int turnedOn)
	{
		const std::vector<int> hours{ 1, 2, 4, 8 };
		const std::vector<int> minutes{ 1, 2, 4, 8, 16, 32 };

		std::vector<std::string> times{};
		backtrack(0, 0, 0, turnedOn, hours, minutes, times);
		return times;

	}
}

// https://leetcode.com/problems/sum-of-left-leaves/
namespace p404
{
	void dfs(TreeNode* node, int& sum, bool isLeft)
	{
		if (!node)
			return;

		if (!node->left && !node->right && isLeft)
			sum += node->val;

		dfs(node->left, sum, true);
		dfs(node->right, sum, false);
	}

	int sumOfLeftLeaves(TreeNode* root)
	{
		int sum = 0;
		dfs(root, sum, false);
		return sum;
	}
}

// https://leetcode.com/problems/convert-a-number-to-hexadecimal/
namespace p405
{
	std::string toHex(int num)
	{
		unsigned int n = static_cast<unsigned int>(num);
		const char hexCodes[]{ '0', '1', '2', '3', '4' , '5' , '6' , '7' , '8' , '9' , 'a' , 'b' , 'c' , 'd' , 'e' , 'f' };

		std::string hex{};
		while (n != 0)
		{
			int remainder = n % 16;
			n /= 16;

			hex += hexCodes[remainder];
		}

		std::reverse(hex.begin(), hex.end());
		return hex;
	}
}

// https://leetcode.com/problems/third-maximum-number/
namespace p414
{
	int thirdMax(const std::vector<int>& nums)
	{
		int firstMaxIdx = 0;
		int secondMaxIdx = -1;
		int thirdMaxIdx = -1;
		for (int i = 1; i < nums.size(); i++)
		{
			if (nums[i] > nums[firstMaxIdx])
			{
				thirdMaxIdx = secondMaxIdx;
				secondMaxIdx = firstMaxIdx;
				firstMaxIdx = i;
			}
			else if (nums[i] != nums[firstMaxIdx])
			{
				if (secondMaxIdx == -1)
				{
					secondMaxIdx = i;
				}
				else if (nums[i] > nums[secondMaxIdx])
				{
					thirdMaxIdx = secondMaxIdx;
					secondMaxIdx = i;
				}
				else if (nums[i] != nums[secondMaxIdx])
				{
					if (thirdMaxIdx == -1 || nums[i] > nums[thirdMaxIdx])
					{
						thirdMaxIdx = i;
					}
				}
			}
		}

		return (thirdMaxIdx != -1) ? nums[thirdMaxIdx] : nums[firstMaxIdx];
	}
}

// https://leetcode.com/problems/add-strings/
namespace p415
{
	std::string addStrings(const std::string& num1, const std::string& num2)
	{
		std::string rev1{ num1 };
		std::string rev2{ num2 };
		const int length1 = rev1.length();
		const int length2 = rev2.length();

		std::reverse(rev1.begin(), rev1.end());
		std::reverse(rev2.begin(), rev2.end());

		std::string result = (length1 >= length2) ? rev1 : rev2;
		std::string other = (length1 >= length2) ? rev2 : rev1;
		int i{};
		int carry = 0;

		// Perform addition using the two numbers
		for (i = 0; i < other.length(); i++)
		{
			int a = result[i] - '0';
			int b = other[i] - '0';
			int sum = a + b + carry;
			carry = sum >= 10;

			result[i] = sum - carry * 10 + '0';
		}

		// Add carry to rest of existing digits
		while (i < result.length())
		{
			int sum = result[i] + carry - '0';
			carry = sum >= 10;
			result[i] = sum - carry * 10 + '0';

			++i;
		}

		if (carry)
			result.push_back('1');

		std::reverse(result.begin(), result.end());
		return result;
	}
}

// https://leetcode.com/problems/number-of-segments-in-a-string/
namespace p434
{
	int countSegments(const std::string& str)
	{
		const int length = str.length();

		int i = 0;
		int numSegments = 0;
		while (i < length)
		{
			// Skip spaces
			while (i < length && str[i] == ' ')
				++i;

			// Non-space char found
			if (i < length)
				++numSegments;

			// Skip until end of segment
			while (i < length && str[i] != ' ')
				++i;
		}

		return numSegments;
	}
}

// https://leetcode.com/problems/arranging-coins/
namespace p441
{
	// Solution 1: brute force
	int arrangeCoins(int n)
	{
		int numCompleteRows = 0;
		int coinsPerRow = 1;
		while (n >= 0)
		{
			n -= coinsPerRow;
			++coinsPerRow;
			++numCompleteRows;
		}

		return numCompleteRows;
	}

	// Solution 2: math formula
	int arrangeCoins2(int n)
	{
		return static_cast<int>(std::sqrt(2.0 * n + 0.25) - 0.5);
	}

	// Solution 3: bit manipulation
	int arrangeCoins3(int n)
	{
		int mask = 1 << 15;
		long result = 0;
		while (mask != 0)
		{
			result |= mask;
			if (result * (result + 1) / 2 > n)
				result ^= mask;
			mask >>= 1;
		}

		return result;
	}

	// Solution 4: binary search
	int arrangeCoins4(int n)
	{
		int left = 0;
		int right = n;
		while (left < right)
		{
			long mid = left + (right - left) / 2;
			long coins = mid * (mid + 1) / 2;
			if (coins == n)
				return static_cast<int>(coins);
			else if (coins > n)
				right = mid - 1;
			else
				left = mid + 1;
		}

		return 0;
	}
}

// https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/
namespace p448
{
	// Solution 1: hashmap
	std::vector<int> findDisappearedNumbers(const std::vector<int>& nums)
	{
		std::unordered_map<int, bool> seen{};
		for (const auto n : nums)
			seen[n] = true;

		std::vector<int> disappeared{};
		for (int i = 1; i <= nums.size(); i++)
			if (!seen[i])
				disappeared.push_back(i);

		return disappeared;
	}

	// Solution 2: indexing tricks
	std::vector<int> findDisappearedNumbers2(std::vector<int>& nums)
	{
		const int n = nums.size();
		for (int i = 0; i < n; i++)
		{
			int idx = nums[i] > 0 ? nums[i] - 1 : nums[i] + n - 1;
			if (nums[idx] > 0)
				nums[idx] -= n;
		}

		std::vector<int> disappeared{};
		for (int i = 0; i < n; i++)
			if (nums[i] > 0)
				disappeared.push_back(i + 1);

		return disappeared;
	}

	// Solution 3: abs
	std::vector<int> findDisappearedNumbers3(std::vector<int>& nums)
	{
		for (const auto num : nums)
		{
			int idx = std::abs(num) - 1;
			nums[idx] = -std::abs(nums[idx]);
		}

		std::vector<int> disappeared{};
		for (int i = 0; i < nums.size(); i++)
			if (nums[i] > 0)
				disappeared.push_back(i + 1);

		return disappeared;
	}
}

// https://leetcode.com/problems/assign-cookies/
namespace p455
{
	// Solution 1: sorting
	int findContentChildren(std::vector<int>& greeds, std::vector<int>& sizes)
	{
		std::sort(greeds.begin(), greeds.end());
		std::sort(sizes.begin(), sizes.end());

		int i = 0;
		int j = 0;
		int content = 0;
		while (i < greeds.size() && j < sizes.size())
		{
			if (sizes[j] >= greeds[i])
			{
				++content;
				++i;
				++j;
			}
			else
			{
				++j;
			}
		}

		return content;
	}
}

// https://leetcode.com/problems/repeated-substring-pattern/
namespace p459
{
	// Solution 1: while loops
	bool repeatedSubstringPattern(const std::string& str)
	{
		const int length = str.length();
		const int halfLength = length / 2;

		char first = str[0];
		int patternLength = 1;
		while (patternLength <= halfLength)
		{
			while (patternLength <= halfLength && str[patternLength] != first)
				++patternLength;

			std::string pattern = str.substr(0, patternLength);
			if (patternLength != length && length % patternLength == 0)	// is candidate valid
			{
				int i = patternLength;
				while (i <= length - patternLength && str.substr(i, patternLength) == pattern)
					i += patternLength;

				if (i == length)
					return true;
			}
			++patternLength;
		}

		return false;
	}

	// Solution 2: starting at half length
	bool repeatedSubstringPattern2(const std::string& str)
	{
		const int len = str.length();
		for (int i = len / 2; i >= 1; --i)
			if (len % i == 0 && str.substr(0, len - i) == str.substr(i))
				return true;

		return false;
	}
}

// https://leetcode.com/problems/hamming-distance/
namespace p461
{
	int countBits(int n)
	{
		int count = 0;
		while (n != 0)
		{
			n &= (n - 1);
			++count;
		}

		return count;
	}

	int hammingDistance(int x, int y)
	{
		int diff = x ^ y;
		return countBits(diff);
	}
}

// https://leetcode.com/problems/island-perimeter/
namespace p463
{
	int countLandNeighbors(const std::vector<std::vector<int>>& grid, int row, int col)
	{
		const std::pair<int, int> neighborCells[]{
			{row - 1, col + 0},
			{row + 0, col - 1},
			{row + 0, col + 1},
			{row + 1, col + 0}
		};

		int rows = grid.size();
		int cols = grid[0].size();
		int count = 0;
		for (const auto& n : neighborCells)
		{
			if (n.first >= 0
				&& n.first < rows
				&& n.second >= 0
				&& n.second < cols)
			{
				count += grid[n.first][n.second];	// 1 for land, 0 for water
			}
		}

		return count;
	}

	// Solution 1: counting all neighbors
	int islandPerimeter(const std::vector<std::vector<int>>& grid)
	{
		int perimeter = 0;
		for (int r = 0; r < grid.size(); r++)
		{
			for (int c = 0; c < grid[0].size(); c++)
			{
				if (grid[r][c] == 1)
				{
					// Land
					perimeter += 4 - countLandNeighbors(grid, r, c);
				}
			}

		}

		return perimeter;
	}

	int islandPerimeter2(const std::vector<std::vector<int>>& grid)
	{
		int count = 0;
		int repeat = 0;
		for (int i = 0; i < grid.size(); i++)
		{
			for (int j = 0; j < grid[0].size(); j++)
			{
				if (grid[i][j] == 1)
				{
					++count;
					if (i != 0 && grid[i - 1][j] == 1)
						++repeat;
					if (j != 0 && grid[i][j - 1] == 1)
						++repeat;
				}
			}
		}

		return 4 * count - 2 * repeat;
	}
}

// https://leetcode.com/problems/number-complement/
namespace p476
{
	int findComplement(int num)
	{
		unsigned int mask = 1;
		while ((num & mask) != num)
			mask |= (mask << 1);

		return num ^ mask;
	}
}

// https://leetcode.com/problems/license-key-formatting/
namespace p482
{
	// Solution 1: insert dash every on every k-th index
	std::string licenseKeyFormatting(const std::string& str, int k)
	{
		std::string formatted{};
		formatted.reserve(str.length());
		for (auto& c : str)
			if (c != '-')
				formatted.push_back(std::toupper(c));


		for (int i = formatted.length() - k; i > 0; i -= k)
			formatted.insert(formatted.begin() + i, '-');

		return formatted;
	}

	// Solution 2: optimized
	std::string licenseKeyFormatting2(const std::string& str, int k)
	{
		std::string alphanumOnly;
		alphanumOnly.reserve(str.length());
		for (auto& c : str)
			if (c != '-')
				alphanumOnly.push_back(std::toupper(c));

		if (alphanumOnly.length() == 0)
			return "";
		const int dashes = (alphanumOnly.length() % k == 0) ? alphanumOnly.length() / k - 1 : alphanumOnly.length() / k;
		const int length = alphanumOnly.length() + dashes;

		std::string formatted{};
		formatted.resize(length);
		int j = alphanumOnly.length() - 1;
		for (int i = length - 1; i >= 0; --i)
		{
			if ((length - i) % (k + 1) == 0)
			{
				formatted[i] = '-';
			}
			else
			{
				formatted[i] = alphanumOnly[j];
				--j;
			}
		}

		return formatted;
	}
}

// https://leetcode.com/problems/max-consecutive-ones/
namespace p485
{
	// Solution 1: iterate through whole array
	int findMaxConsecutiveOnes(const std::vector<int>& nums)
	{
		const int size = nums.size();
		int max = 0;
		int idx = 0;
		for (int i = 0; i < size; i++)
		{
			if (nums[i] == 0)
			{
				max = std::max(max, i - idx);
				idx = i + 1;
			}
		}

		max = std::max(max, size - idx);
		return max;
	}
}

// https://leetcode.com/problems/construct-the-rectangle/
namespace p492
{
	// solution 1: loop through all factors
	std::vector<int> constructRectangle(int area)
	{
		int length = area;
		int width = 1;
		for (int i = 2; i <= std::sqrt(area); i++)
		{
			if (area % i == 0)
			{
				width = i;
				length = area / i;
			}
		}

		return { length, width };
	}

	// Solution 2: start from square root
	std::vector<int> constructRectangle2(int area)
	{
		for (int i = std::sqrt(area); i >= 1; --i)
			if (area % i == 0)
				return { area / i, i };

		return { area, 1 };	// should not be reached
	}
}

// https://leetcode.com/problems/teemo-attacking/
namespace p495
{
	int findPoisonedDuration(const std::vector<int>& timeSeries, int duration)
	{
		int total = 0;
		for (size_t i = 0; i < timeSeries.size() - 1; i++)
		{
			if (timeSeries[i] + duration <= timeSeries[i + 1])
				total += duration;
			else
				total += timeSeries[i + 1] - timeSeries[i];
		}

		total += duration;
		return total;
	}
}

// https://leetcode.com/problems/next-greater-element-i/
namespace p496
{
	// Solution 1: using a single map for storing indices only
	std::vector<int> nextGreaterElement(const std::vector<int>& nums1, const std::vector<int>& nums2)
	{
		std::unordered_map<int, int> map{};
		for (int j = 0; j < nums2.size(); j++)
			map[nums2[j]] = j;

		std::vector<int> res{};
		for (int i = 0; i < nums1.size(); i++)
		{
			int j = map[nums1[i]];
			int x = nums2[j];
			int greater = -1;
			while (j + 1 < nums2.size())
			{
				if (nums2[j + 1] > x)
				{
					greater = nums2[j + 1];
					break;
				}
				++j;
			}

			res.push_back(greater);
		}

		return res;
	}

	// Solution 2: monotonic stack
	std::vector<int> nextGreaterElement2(const std::vector<int>& nums1, const std::vector<int>& nums2)
	{
		std::unordered_map<int, int> map{};
		std::stack<int> stack{};

		for (const auto n : nums2)
		{
			while (!stack.empty() && n > stack.top())
			{
				int top = stack.top();
				map[top] = n;
				stack.pop();
			}
			stack.push(n);
		}

		std::vector<int> res{};
		res.reserve(nums1.size());
		for (const auto n : nums1)
		{
			auto it = map.find(n);
			if (it != map.end())
				res.push_back(it->second);
			else
				res.push_back(-1);
		}

		return res;
	}
}

// https://leetcode.com/problems/keyboard-row/
namespace p500
{
	std::vector<std::string> findWords(const std::vector<std::string>& words)
	{
		// Fill keyboard
		std::vector<std::unordered_map<char, int>> keyboard{};
		keyboard.reserve(3);
		for (const std::string& row : { "qwertyuiop", "asdfghjkl", "zxcvbnm" })
		{
			std::unordered_map<char, int> rowMap{};
			for (const auto c : row)
				rowMap[c] = 1;
			keyboard.push_back(rowMap);
		}

		// Check a word with each of the rows
		std::vector<std::string> res{};
		for (const auto& word : words)
		{
			bool canBeTyped = false;
			for (const auto& rowMap : keyboard)
			{
				canBeTyped = true;
				for (const auto c : word)
				{
					if (rowMap.find(std::tolower(c)) == rowMap.end())
						canBeTyped = false;
				}

				if (canBeTyped)
					res.push_back(word);
			}
		}

		return res;
	}
}

// https://leetcode.com/problems/find-mode-in-binary-search-tree/
namespace p501
{
	void dfs(TreeNode* node, std::unordered_map<int, int>& freq, int& maxFreq)
	{
		if (!node)
			return;

		++freq[node->val];
		maxFreq = std::max(maxFreq, freq[node->val]);
		dfs(node->left, freq, maxFreq);
		dfs(node->right, freq, maxFreq);
	}

	// Solution 1: using hashmap
	std::vector<int> findMode(TreeNode* root)
	{
		std::unordered_map<int, int> freq{};
		int maxFreq = 0;
		dfs(root, freq, maxFreq);

		std::vector<int> res{};
		for (auto it = freq.begin(); it != freq.end(); ++it)
			if (it->second == maxFreq)
				res.push_back(it->first);

		return res;
	}

	void dfs2(TreeNode* node, std::vector<int>& modes, int& mode, int& currFreq, int& maxFreq)
	{
		if (!node)
			return;

		dfs2(node->left, modes, mode, currFreq, maxFreq);

		if (mode == node->val)
		{
			++currFreq;
		}
		else
		{
			mode = node->val;
			currFreq = 1;
		}

		if (currFreq > maxFreq)
		{
			maxFreq = currFreq;
			modes = { mode };
		}
		else if (currFreq == maxFreq)
		{
			modes.push_back(node->val);
		}

		dfs2(node->right, modes, mode, currFreq, maxFreq);
	}

	// Solution 2: inorder
	std::vector<int> findMode2(TreeNode* root)
	{
		std::vector<int> modes{};
		int currFreq = 0;
		int maxFreq = 0;
		int mode = INT_MIN;
		dfs2(root, modes, mode, currFreq, maxFreq);

		return modes;
	}
}

// https://leetcode.com/problems/base-7/
namespace p504
{
	std::string convertToBase7(int num)
	{
		if (num == 0)
			return "0";

		int n = std::abs(num);
		std::string repr{};
		while (n != 0)
		{
			repr += (n % 7 + '0');
			n /= 7;
		}

		if (num < 0)
			repr += '-';

		std::reverse(repr.begin(), repr.end());
		return repr;
	}
}

// https://leetcode.com/problems/relative-ranks/
namespace p506
{
	// Solution 1: sorting
	std::vector<std::string> findRelativeRanks(std::vector<int>& scores)
	{
		const int n = scores.size();
		std::unordered_map<int, int> originalPlaces{};
		originalPlaces.reserve(n);
		for (int i = 0; i < n; i++)
			originalPlaces[scores[i]] = i;

		std::sort(scores.begin(), scores.end(), std::greater<int>());
		std::vector<std::string> ranks(n);
		for (int i = 0; i < n; i++)
		{
			std::string rank{};
			switch (i)
			{
			case 0:
				rank = "Gold Medal";
				break;
			case 1:
				rank = "Silver Medal";
				break;
			case 2:
				rank = "Bronze Medal";
				break;
			default:
				rank = std::to_string(i + 1);
				break;
			}

			ranks[originalPlaces[scores[i]]] = rank;
		}

		return ranks;
	}

	// Solution 2: priority queue
	std::vector<std::string> findRelativeRanks2(std::vector<int>& scores)
	{
		auto toRank = [](int placement) -> std::string {
			switch (placement)
			{
			case 1:		return "Gold Medal";
			case 2:		return "Silver Medal";
			case 3:		return "Bronze Medal";
			default:	return std::to_string(placement);
			}
			};

		std::priority_queue<std::pair<int, int>> pq{};
		const int n = scores.size();
		for (int i = 0; i < n; i++)
			pq.push({ scores[i], i });

		std::vector<std::string> ranks(n);
		for (int i = 1; i <= n; i++)
		{
			ranks[pq.top().second] = toRank(i);
			pq.pop();
		}

		return ranks;
	}
}

// https://leetcode.com/problems/perfect-number/
namespace p507
{
	// Solution 1: checking divisors
	bool checkPerfectNumber(int num)
	{
		int sum = 1;
		for (int i = 2; i * i < num; i++)
		{
			if (num % i == 0)
				sum += i + num / i;
			if (sum > num)
				return false;
		}

		return num != 1 && num == sum;
	}

	// Solution 2: Euclid-Euler theorem
	bool checkPerfectNumber2(int num)
	{
		const long primes[] = { 2, 3, 5, 7, 13, 17, 19, 31 };
		for (const auto p : primes)
			if ((1l << (p - 1l)) * ((1l << p) - 1l) == num)
				return true;
		return false;
	}
}

// https://leetcode.com/problems/fibonacci-number/
namespace p509
{
	// Solution 1: swapping vars
	int fib(int n)
	{
		int a = 0;
		int b = 1;
		while (n != 0)
		{
			int sum = a + b;
			a = b;
			b = sum;
			n--;
		}

		return b;
	}

	// Solution 2: dynamic programming
	int fib2(int n)
	{
		std::vector<int> fibs(n + 2, 0);
		fibs[0] = 0;
		fibs[1] = 1;
		for (int i = 2; i < n + 1; i++)
			fibs[i] = fibs[i - 1] + fibs[i - 2];

		return fibs[n];
	}

	int fibMemo(int n, std::unordered_map<int, int>& cache)
	{
		if (auto it = cache.find(n); it != cache.end())
		{
			return it->second;
		}
		else
		{
			cache[n] = fibMemo(n - 1, cache) + fibMemo(n - 2, cache);
			return cache[n];
		}
	}

	// Solution 3: recursion with memoization
	int fib3(int n)
	{
		std::unordered_map<int, int> cache{};
		cache[0] = 0;
		cache[1] = 1;
		return fibMemo(n, cache);
	}
}

// https://leetcode.com/problems/detect-capital/
// TTS: 30:31
namespace p520
{
	// Solution 1: brute force
	bool detectCapitalUse(const std::string& word)
	{
		bool capital = std::tolower(word[0]) != word[0];
		if (capital && word.length() > 1)
		{
			bool secondCapital = std::tolower(word[1]) != word[1];
			for (int i = 1; i < word.length(); i++)
			{
				if ((std::tolower(word[i]) == word[i]) != secondCapital)
					return false;
			}
		}
		else
		{
			bool secondCapital = false;
			for (int i = 1; i < word.length(); i++)
			{
				if ((std::tolower(word[i]) == word[i]) != secondCapital)
					return false;
			}
		}

		return true;
	}

	// Solution 2: clean
	bool detectCapitalUse(const std::string& word)
	{
		if (word.size() == 1)
			return true;

		int count = 0;
		for (const auto c : word)
			if (std::isupper(c))
				++count;

		if ((count == 1 && std::isupper(word[0]))
			|| (count == 0 || count == word.length()))
		{
			return true;
		}
		return false;
	}
}

// https://leetcode.com/problems/longest-uncommon-subsequence-i/
// TTS: 16:33
namespace p521
{
	// Solution 1: checking substring one-by-one
	int findLUSlength(std::string& a, std::string& b)
	{
		if (a.length() < b.length())
			std::swap(a, b);

		const int aLength = a.length();
		const int bLength = b.length();

		// Substrings of decreasing lengths
		for (int l = aLength; l >= 0; --l)
		{
			if (l > bLength)
				return l;

			for (int j = 0; j <= aLength - l; j++)
			{
				std::string substr = a.substr(j, l);
				bool isSubstr = false;
				for (int k = 0; k <= bLength - l; k++)
				{
					if (b.substr(k, l) == substr)
					{
						isSubstr = true;
						break;
					}
				}

				if (!isSubstr)
					return l;
			}
		}

		return -1;
	}

	// Solution 2: check basic equality
	int findLUSlength(const std::string& a, const std::string& b)
	{
		return (a == b) ? -1 : std::max(a.length(), b.length());
	}
}

// https://leetcode.com/problems/minimum-absolute-difference-in-bst/
// TTS: 11:21
namespace p530
{
	void inorder(TreeNode* node, int& prev, int& diff)
	{
		if (!node)
			return;

		inorder(node->left, node->val, diff);
		std::cout << prev << ' ' << node->val << '\n';
		diff = std::min(diff, std::abs(node->val - prev));
		prev = node->val;
		inorder(node->right, node->val, diff);
	}

	int getMinimumDifference(TreeNode* root)
	{
		int diff = INT_MAX;
		int prev = INT_MAX;
		inorder(root, prev, diff);
		return diff;
	}
}

// https://leetcode.com/problems/reverse-string-ii/
// TTS: 17:39
namespace p541
{
	void reverse(std::string& str, int left, int right)
	{
		while (left < right)
		{
			std::swap(str[left], str[right]);
			++left;
			--right;
		}
	}

	std::string reverseStr(const std::string& s, int k)
	{
		const int length = s.length();
		std::string res = s;
		const int group = 2 * k;
		int i{};
		for (i = 0; i < length; i += group)
		{
			if (i + k - 1 < length)
				reverse(res, i, i + k - 1);
			else
				reverse(res, i, length - 1);
		}

		return res;
	}
}

// https://leetcode.com/problems/diameter-of-binary-tree/
// TTS: 42:41
namespace p543
{
	int dfs(TreeNode* node, int& diameter)
	{
		if (!node)
			return 0;

		int leftHeight = dfs(node->left, diameter);
		int rightHeight = dfs(node->right, diameter);
		diameter = std::max(diameter, leftHeight + rightHeight);
		return std::max(leftHeight, rightHeight) + 1;
	}

	int diameterOfBinaryTree(TreeNode* root)
	{
		int diameter = 0;
		dfs(root, diameter);
		return diameter;
	}
}

// https://leetcode.com/problems/student-attendance-record-i/
namespace p551
{
	// Solution 1: lengthy
	bool checkRecord(const std::string& str)
	{
		int absent = 0;
		for (int i = 0; i < str.length(); i++)
		{
			if (str[i] == 'L')
			{
				int j = i;
				while (j < str.length() && str[j] == 'L')
					++j;

				if (j - i >= 3)
					return false;

				i = j - 1;
			}
			else if (str[i] == 'A')
			{
				++absent;
				if (absent >= 2)
					return false;
			}
		}

		return true;
	}

	// Solution 2: concise
	bool checkRecord2(const std::string& str)
	{
		int absent = 0;
		int late = 0;
		for (const auto c : str)
		{
			if (c == 'A')
				++absent;
			if (c == 'L')
				++late;
			else
				late = 0;

			if (absent >= 2 || late >= 3)
				return false;
		}

		return true;
	}
}

// https://leetcode.com/problems/maximum-depth-of-n-ary-tree/
// TTS: 12:14
namespace p559
{
	void dfs(Node* node, int& maxDepth, int currDepth = 1)
	{
		if (!node)
			return;

		for (auto* n : node->children)
			dfs(n, maxDepth, currDepth + 1);

		maxDepth = std::max(maxDepth, currDepth);
	}

	// Solution 1: mine
	int maxDepth(Node* root)
	{
		int maxDepth = 0;
		dfs(root, maxDepth);
		return maxDepth;
	}

	// Solution 2: Recursive DFS
	int maxDepth2(Node* root)
	{
		if (!root)
			return 0;

		int depth = 0;
		for (auto* child : root->children)
			depth = std::max(depth, maxDepth2(child));

		return depth + 1;
	}

	// Solution 3: Recursive DFS using STL
	int maxDepth3(Node* root)
	{
		if (!root)
			return 0;

		return 1 + std::transform_reduce(root->children.begin(), root->children.end(), 0,
			[](int acc, int d) {return std::max(acc, d);}, [](auto* node) {return maxDepth3(node);});
	}

	// Solution 4: BFS
	int maxDepth4(Node* root)
	{
		if (!root)
			return 0;

		std::queue<Node*> queue{};
		queue.push(root);
		int depth = 0;
		while (!queue.empty())
		{
			depth += 1;
			int breadth = queue.size();
			for (int i = 0; i < breadth; i++)
			{
				auto* node = queue.front();
				queue.pop();
				for (auto* child : node->children)
					if (child)
						queue.push(child);
			}
		}

		return depth;
	}
}

// https://leetcode.com/problems/array-partition/
// TTS: 03:30
namespace p561
{
	// Solution 1: sort
	int arrayPairSum(std::vector<int>& nums)
	{
		std::sort(nums.begin(), nums.end(), std::greater<int>());
		int sum = 0;
		for (int i = 0; i < nums.size() - 1; i += 2)
			sum += std::min(nums[i], nums[i + 1]);

		return sum;
	}

	// Solution 2: sort 2
	int arrayPairSum2(std::vector<int>& nums)
	{
		std::sort(nums.begin(), nums.end(), std::greater<int>());
		int sum = 0;
		for (int i = 1; i < nums.size(); i += 2)
			sum += nums[i];

		return sum;
	}

	// Solution 3: bucket sort / frequency counting knowing the range beforehand
	int arrayPairSum3(const std::vector<int>& nums)
	{
		int* buckets = new int[20001] {};
		for (const auto n : nums)
			++buckets[n + 10000];	// range starts from -10000 (kinda bad :( )

		int sum = 0;
		int used = 0;
		for (int i = 0; used < nums.size(); )
		{
			if (buckets[i] == 0)
			{
				++i;
			}
			else
			{
				if (used % 2 == 0)
					sum += i - 10000;

				--buckets[i];
				++used;
			}
		}

		return sum;
	}
}

// https://leetcode.com/problems/binary-tree-tilt/
// TTS: 12:14
namespace p563
{
	int dfs(TreeNode* node, int& tiltSum)
	{
		if (!node)
			return 0;

		int leftSum = dfs(node->left, tiltSum);
		int rightSum = dfs(node->right, tiltSum);
		tiltSum += std::abs(rightSum - leftSum);
		return leftSum + rightSum + node->val;
	}

	int findTilt(TreeNode* root)
	{
		int res = 0;
		dfs(root, res);
		return res;
	}
}

// https://leetcode.com/problems/reshape-the-matrix/
// TTS: 08:10
namespace p566
{
	// Solution 1: 4 indices
	std::vector<std::vector<int>> matrixReshape(const std::vector<std::vector<int>>& mat, int r, int c)
	{
		const int m = mat.size();
		const int n = mat[0].size();
		if (r * c != m * n)
			return mat;	// invalid size

		int i = 0;
		int j = 0;
		std::vector<std::vector<int>> reshaped(r, std::vector<int>(c));
		for (int k = 0; k < m; k++)
		{
			for (int l = 0; l < n; l++)
			{
				// If end of row, go to next row and reset col index
				if (j >= c)
				{
					++i;
					j = 0;
				}

				reshaped[i][j] = mat[k][l];
				++j;
			}

		}

		return reshaped;
	}

	// Solution 2: clean loop
	std::vector<std::vector<int>> matrixReshape2(const std::vector<std::vector<int>>& mat, int r, int c)
	{
		const int m = mat.size();
		const int n = mat[0].size();
		if (r * c != m * n)
			return mat;	// invalid size

		std::vector<std::vector<int>> reshaped(r, std::vector<int>(c));
		for (int i = 0; i < r * c; i++)
			reshaped[i / c][i % c] = mat[i / n][i % n];

		return reshaped;
	}
}

// https://leetcode.com/problems/subtree-of-another-tree/
// TTS: a lot
namespace p572
{
	void dfs(TreeNode* node, std::string& traversal)
	{
		if (!node)
		{
			traversal += " ";
			return;
		}

		traversal += std::to_string(node->val);
		dfs(node->left, traversal);
		dfs(node->right, traversal);
	}

	// Solution 1: storing traversal
	bool isSubtree(TreeNode* root, TreeNode* subRoot)
	{
		std::string t1{};
		std::string t2{};
		dfs(root, t1);
		dfs(subRoot, t2);

		for (int i = 0; i < t1.length(); i++)
		{
			int j = i;
			while (j < t2.length() && t1[j] == t2[j])
				++j;

			if (j - i == t2.length())
				return true;
		}

		return false;
	}

	int getDepth(TreeNode* node, std::vector<TreeNode*>& nodes, int depth)
	{
		if (!node)
			return -1;

		int maxDepth = std::max(getDepth(node->left, nodes, depth), getDepth(node->right, nodes, depth)) + 1;
		if (maxDepth == depth)
			nodes.push_back(node);

		return maxDepth;
	}

	bool identical(TreeNode* t1, TreeNode* t2)
	{
		if (!t1 && !t2)
			return true;
		if (!t1 || !t2 || t1->val != t2->val)
			return false;

		return identical(t1->left, t2->left) && identical(t1->right, t2->right);
	}

	// Solution 2: Checking identical trees
	bool isSubtree2(TreeNode* t1, TreeNode* t2)
	{
		if (!t1 && !t2)
			return true;
		if (!t1 || !t2)
			return false;

		std::vector<TreeNode*> t1Nodes{};
		int t2Depth = getDepth(t2, t1Nodes, -1);
		int t1Depth = getDepth(t1, t1Nodes, t2Depth);

		for (TreeNode* n : t1Nodes)
			if (identical(n, t2))
				return true;

		return false;
	}

	std::string serialize(TreeNode* t)
	{
		if (!t)
			return ",#";

		return "," + std::to_string(t->val) + serialize(t->left) + serialize(t->right);
	}

	std::vector<int> getLPS(const std::string& str)
	{
		const int m = str.length();
		int j = 0;
		std::vector<int> lps(m);
		for (int i = 1; i < m; i++)
		{
			while (str[i] != str[j] && j > 0)
				j = lps[j];
			if (str[i] == str[j])
			{
				++j;
				lps[i] = j;
			}
		}

		return lps;
	}

	bool kmp(const std::string& str, const std::string& pattern)
	{
		std::vector<int> lps = getLPS(pattern);
		const int n = str.length();
		const int m = pattern.length();
		int j = 0;
		for (int i = 0; i < n; i++)
		{
			while (str[i] != pattern[j] && j > 0)
				j = lps[j - 1];
			if (str[i] == pattern[j])
			{
				++j;
				if (j == m)
					return true;
			}
		}

		return false;
	}

	// Solution 3: DFS + KMP
	bool isSubtree3(TreeNode* t1, TreeNode* t2)
	{
		return kmp(serialize(t1), serialize(t2));
	}
}

// https://leetcode.com/problems/distribute-candies/
// TTS: 06:10
namespace p575
{
	// Solution 1: hashmap
	int distributeCandies(const std::vector<int>& candyType)
	{
		std::unordered_map<int, int> map{};
		for (const auto n : candyType)
			++map[n];

		int uniqueCandies = 0;
		for (auto it = map.begin(); it != map.end(); ++it)
			++uniqueCandies;

		return std::min(uniqueCandies, static_cast<int>(candyType.size() / 2));
	}

	// Solution 2: sort
	int distributeCandies2(std::vector<int>& candyType)
	{
		std::sort(candyType.begin(), candyType.end());
		const int n = candyType.size();

		int uniqueCandies = 1;
		for (int i = 1; i < n; i++)
			if (candyType[i] != candyType[i - 1])
				++uniqueCandies;

		return std::min(uniqueCandies, static_cast<int>(n / 2));
	}

	// Solution 3: unordered set
	int distributeCandies3(const std::vector<int>& candyType)
	{
		std::unordered_set<int> set(candyType.begin(), candyType.end());
		return std::min(set.size(), candyType.size() / 2);
	}
}

// https://leetcode.com/problems/n-ary-tree-preorder-traversal/
// TTS: 07:33
namespace p589
{
	void dfs(Node* node, std::vector<int>& traversal)
	{
		if (!node)
			return;

		traversal.push_back(node->val);
		for (auto child : node->children)
			dfs(child, traversal);
	}

	std::vector<int> preorder(Node* root)
	{
		std::vector<int> res{};
		dfs(root, res);
		return res;
	}
}

// https://leetcode.com/problems/n-ary-tree-postorder-traversal/
// TTS: 01:29
namespace p590
{
	void dfs(Node* node, std::vector<int>& traversal)
	{
		if (!node)
			return;

		for (auto child : node->children)
			dfs(child, traversal);

		traversal.push_back(node->val);
	}

	std::vector<int> postorder(Node* root)
	{
		std::vector<int> res{};
		dfs(root, res);
		return res;
	}
}

// https://leetcode.com/problems/longest-harmonious-subsequence/
// TTS: 43:45
namespace p594
{
	// Solution 1: sorting
	int findLHS(std::vector<int>& nums)
	{
		std::sort(nums.begin(), nums.end());

		int i = 0;
		int longest = 0;
		while (i < nums.size())
		{
			int count = 1;
			while (i < nums.size() - 1 && nums[i] == nums[i + 1])
			{
				++i;
				++count;
			}

			int j = i + 1;
			while (j < nums.size() && std::abs(nums[i] - nums[j]) == 1)
			{
				++j;
				++count;
			}

			if (j != i + 1)
				longest = std::max(longest, count);
			++i;
		}

		return longest;
	}

	// Solution 2: sorting v2
	int findLHS(std::vector<int>& nums)
	{
		std::sort(nums.begin(), nums.end());

		int longest = 0;
		int left = 0;
		for (int right = 0; right < nums.size(); ++right)
		{
			while (nums[right] - nums[left] > 1)
				++left;

			if (nums[right] - nums[left] == 1)
				longest = std::max(longest, right - left + 1);
		}

		return longest;
	}

	// Solution 2: hashmap
	int findLHS(const std::vector<int>& nums)
	{
		std::unordered_map<int, int> map{};
		for (auto n : nums)
			++map[n];

		int longest = 0;
		for (auto it = map.begin(); it != map.end(); ++it)
			if (auto it2 = map.find(it->first + 1); it2 != map.end())
				longest = std::max(longest, it->second + it2->second);

		return longest;
	}
}

// https://leetcode.com/problems/range-addition-ii/
// TTS: 35:31
namespace p598
{
	int maxCount(int m, int n, const std::vector<std::vector<int>>& ops)
	{
		int minRow = m;
		int minCol = n;
		for (const auto& op : ops)
		{
			minRow = std::min(minRow, op[0]);
			minCol = std::min(minCol, op[1]);
		}

		return minRow * minCol;
	}
}

// https://leetcode.com/problems/minimum-index-sum-of-two-lists/
// TTS: 14:38
namespace p599
{
	// Solution 1: sorting and hashmap
	std::vector<std::string> findRestaurant(std::vector<std::string>& list1, std::vector<std::string>& list2)
	{
		std::unordered_map<std::string, int> indexMap1{};
		std::unordered_map<std::string, int> indexMap2{};
		for (int i = 0; i < list1.size(); i++)
			indexMap1[list1[i]] = i;
		for (int i = 0; i < list2.size(); i++)
			indexMap2[list2[i]] = i;

		std::sort(list1.begin(), list1.end());
		std::sort(list2.begin(), list2.end());

		int i = 0;
		int j = 0;
		int minIdxSum = list1.size() + list2.size();
		std::vector<std::string> commonStrings{};
		while (i < list1.size() && j < list2.size())
		{
			if (list1[i] == list2[j])
			{
				if (indexMap1[list1[i]] + indexMap2[list2[j]] == minIdxSum)
				{
					commonStrings.push_back(list1[i]);
				}
				else if (indexMap1[list1[i]] + indexMap2[list2[j]] < minIdxSum)
				{
					minIdxSum = indexMap1[list1[i]] + indexMap2[list2[j]];
					commonStrings = { list1[i] };
				}

				++i;
				++j;
			}
			else if (list1[i] < list2[j])
			{
				++i;
			}
			else
			{
				++j;
			}
		}

		return commonStrings;
	}

	// Solution 2: sorting and hashmap
	std::vector<std::string> findRestaurant(const std::vector<std::string>& list1, const std::vector<std::string>& list2)
	{
		std::unordered_map<std::string, int> map{};
		const int size1 = list1.size();
		const int size2 = list2.size();
		for (int i = 0; i < size1; i++)
			map[list1[i]] = i;

		std::vector<std::string> res{};
		int min = size1 + size2;
		for (int i = 0; i < size2; i++)
		{
			if (auto it = map.find(list2[i]);it != map.end())
			{
				if (i + it->second == min)
				{
					res.clear();
					res.push_back(list2[i]);
				}
				else if (i + it->second < min)
				{
					min = i + it->second;
					res = { list2[i] };
				}
			}
		}

		return res;
	}
}

// https://leetcode.com/problems/can-place-flowers/
// TTS: 37:14
namespace p605
{
	bool canPlaceFlowers(std::vector<int>& flowerbed, int n)
	{
		flowerbed.insert(flowerbed.begin(), 0);
		flowerbed.push_back(0);

		for (int i = 1; i < flowerbed.size() - 1; i++)
		{
			if (flowerbed[i - 1] == 0 && flowerbed[i] == 0 && flowerbed[i + 1] == 0)
			{
				flowerbed[i] == 1;
				--n;
				++i;
			}
		}

		return n <= 0;
	}
}

// https://leetcode.com/problems/merge-two-binary-trees/
// TTS: 17:51
namespace p617
{
	void mergeDFS(TreeNode* t1, TreeNode* t2, TreeNode*& merged)
	{
		if (!t1 || !t2)
		{
			merged = t1 ? t1 : t2;
			return;
		}

		merged = new TreeNode(t1->val + t2->val);
		mergeDFS(t1->left, t2->left, merged->left);
		mergeDFS(t1->right, t2->right, merged->right);
	}

	TreeNode* mergeTrees(TreeNode* root1, TreeNode* root2)
	{
		TreeNode* merged{};
		mergeDFS(root1, root2, merged);
		return merged;
	}

	// Solution 2: simpler
	TreeNode* mergeTrees2(TreeNode* root1, TreeNode* root2)
	{
		if (!root1 || !root2)
			return root1 ? root1 : root2;

		root1->val += root2->val;
		root1->left = mergeTrees2(root1->left, root2->left);
		root1->right = mergeTrees2(root1->right, root2->right);
		return root1;
	}
}

// https://leetcode.com/problems/maximum-product-of-three-numbers/
// TTS:
namespace p628
{
	// Solution 1: sorting
	int maximumProduct(std::vector<int>& nums)
	{
		std::sort(nums.begin(), nums.end());

		const int size = nums.size();
		int prod1 = nums[0] * nums[1] * nums[size - 1];
		int prod2 = nums[size - 3] * nums[size - 2] * nums[size - 1];
		return std::max(prod1, prod2);
	}

	// Solution 2: keeping track of maxes and  mins
	int maximumProduct(std::vector<int>& nums)
	{
		int max1 = INT_MIN;
		int max2 = INT_MIN;
		int max3 = INT_MIN;

		int min1 = INT_MAX;
		int min2 = INT_MAX;

		for (int i = 0; i < nums.size(); i++)
		{
			if (nums[i] <= min1)
			{
				min2 = min1;
				min1 = nums[i];
			}
			else if (nums[i] <= min2)
			{
				min2 = nums[i];
			}

			if (nums[i] >= max1)
			{
				max3 = max2;
				max2 = max1;
				max1 = nums[i];
			}
			else if (nums[i] >= max2)
			{
				max3 = max2;
				max2 = nums[i];
			}
			else if (nums[i] >= max3)
			{
				max3 = nums[i];
			}
		}

		return std::max(min1 * min2 * max1, max1 * max2 * max3);
	}
}

int main()
{

}

