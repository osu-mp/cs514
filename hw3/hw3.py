"""
Matthew Pacey
CS 514 - HW3 - Heapsort
"""

import math
import unittest


def build_max_heap(nums):
    """
    Build a max heap (in place) from the input number list
    """
    n = len(nums)
    # the last parent is the floor of the midpoint of the array
    # this is the last node with children, so work backwards from it
    last_parent = n // 2 - 1
    for i in range(last_parent, -1, -1):
        max_heapify(nums, n, i)


def max_heapify(nums, heap_size, i):
    """
    Build a max heap of the input number list starting from the ith element
    Ignore any elements after heap-size as heapSort will incrementally
    fill the list with the largest numbers at the back, making the target
    heap smaller and smaller.
    """
    # print(f"max_heapify {nums=}, {heap_size=}, {i=}")
    largest = i  # index of the root node we are considering
    left = 2 * i + 1  # left child index
    right = 2 * i + 2  # right child index

    # if left exists and is larger than the root, it needs to bubble up
    if left < heap_size and nums[left] > nums[largest]:
        largest = left

    # similarly if right exists and is larger than the root, it needs to bubble up
    if right < heap_size and nums[right] > nums[largest]:
        largest = right

    # if either of the children are larger than the root, swap them out
    # and recursively max_heapify the current tree until no swaps occur
    if largest != i:
        nums[i], nums[largest] = nums[largest], nums[i]  # Swap
        max_heapify(nums, heap_size, largest)


def heapSort(nums):
    """
    Builds a max-heap from the input array
    Starting with the root (the maximum element), the algorithm places the maximum element into
    the correct place in the array by swapping it with the element in the last position in the array.
    Discard this last node (knowing that it is in its correct place) by decreasing the heap size, and
    calling MAX-HEAPIFY on the new (possibly incorrectly-placed) root.
    Repeat this discarding process until only one node (the smallest element) remains, and
    therefore is in the correct place in the array.
    """
    # print(f"heapSort start {nums=}")
    # build a max heap from entire list
    build_max_heap(nums)
    # print(f"heapSort max_heap {nums=}")
    last = len(nums) - 1
    for i in range(last, 0, -1):
        # swap the root value (max) to the end of the list
        nums[i], nums[0] = nums[0], nums[i]
        # re-heapify the list, ignoring the last i values
        max_heapify(nums, i, 0)

    # print(f"heapSort sorted {nums=}")
    return nums


class minProrityQueue:
    def __init__(self, min_heap):
        self.min_heap = min_heap

    def insert(self, priority):
        """
        Insert an element with the given priority
        """
        self.min_heap.append(priority)
        # print(f"{self.min_heap=}")
        child_i = len(self.min_heap) - 1
        parent_i = (child_i - 1) // 2
        # print(f"{parent_i=}, {self.min_heap[parent_i]=}")
        while self.min_heap[child_i] < self.min_heap[parent_i]:
            # print(f"{child_i=}, {self.min_heap[child_i]=}, < {parent_i=}, {self.min_heap[parent_i]}")
            self.min_heap[child_i], self.min_heap[parent_i] = self.min_heap[parent_i], self.min_heap[child_i]
            child_i = parent_i
            parent_i = (child_i - 1) // 2
            # print(f"next {parent_i=}, {child_i=}")

            if parent_i < 0:
                break

        # print(f"final after insert {self.min_heap=}")

    def first(self):
        """
        Return the element with the lowest/minimum priority value
        This is just the first element in the array
        """
        if len(self.min_heap) == 0:
            return None
        return self.min_heap[0]

    def remove_first(self):
        """
        Pop and return the element with the lowest/minimum priority
        """
        # save the value of the first index for later return
        first = self.first()

        if len(self.min_heap) == 0:
            return None

        # move the last element to the first element (breaks heap, will be fixed)
        self.min_heap[0] = self.min_heap[-1]
        # remove last element from heap
        del self.min_heap[-1]

        i = 0
        heap_size = len(self.min_heap)
        while True:
            left = i * 2 + 1
            right = i * 2 + 2
            smallest = i

            # find the min of the left and right children (if they exist)
            if left < heap_size and self.min_heap[left] < self.min_heap[smallest]:
                smallest = left
            if right < heap_size and self.min_heap[right] < self.min_heap[smallest]:
                smallest = right

            # min value is not going to bubble down anymore
            if smallest == i:
                break

            # else, buble down the value to the smaller
            self.min_heap[i], self.min_heap[smallest] = self.min_heap[smallest], self.min_heap[i]
            i = smallest

        return first


class Testing(unittest.TestCase):
    def test_max_heap(self):
        nums = [10, 1, 27, 29, 33, 4, 66, 2, 5]
        self.assertEqual(heapSort(nums), sorted(nums))

        nums = list(range(0, 10))
        self.assertEqual(heapSort(nums), sorted(nums))

        nums = list(range(10, 0, -1))
        self.assertEqual(heapSort(nums), sorted(nums))

    def test_min_heap(self):
        nums = [32, 12, 2, 8, 16, 20, 24, 40, 4]
        expected = [2, 4, 20, 8, 16, 32, 24, 40, 12]
        build_min_heap(nums)
        self.assertEqual(nums, expected)

    def test_queue(self):
        nums = [32, 12, 2, 8, 16, 20, 24, 40, 4]
        build_min_heap(nums)
        queue = minProrityQueue(nums)

        self.assertEqual(queue.first(), 2)
        first = queue.remove_first()
        self.assertEqual(first, 2)
        first = queue.remove_first()
        self.assertEqual(first, 4)
        first = queue.remove_first()
        self.assertEqual(first, 8)

        nums_copy = queue.min_heap.copy()
        nums_copy.append(3)
        queue.insert(3)
        build_min_heap(nums_copy)
        self.assertEqual(queue.min_heap, nums_copy)

        nums_copy = queue.min_heap.copy()
        nums_copy.append(17)
        queue.insert(17)
        build_min_heap(nums_copy)
        self.assertEqual(queue.min_heap, nums_copy)

        nums_copy = queue.min_heap.copy()
        nums_copy.append(9)
        queue.insert(9)
        build_min_heap(nums_copy)
        self.assertEqual(queue.min_heap, nums_copy)

        nums_copy = queue.min_heap.copy()
        nums_copy.append(7)
        queue.insert(7)
        build_min_heap(nums_copy)
        self.assertEqual(queue.min_heap, nums_copy)

    def test_empty_queue(self):
        nums = []
        queue = minProrityQueue(nums)
        self.assertEqual(queue.min_heap, [])
        first = queue.remove_first()
        self.assertEqual(first, None)
        queue.insert(3)
        first = queue.remove_first()
        self.assertEqual(first, 3)
        first = queue.remove_first()
        self.assertEqual(first, None)


# create min heaps as almost a mirror image of max heap
# these funcs allow for easier testing of minPriorityQueue funcs
# as the minPriorityQueue funcs are essentially restoring the min-heapness of the member var
def build_min_heap(nums):
    """
    Build a min heap (in place) from the input number list
    """
    n = len(nums)
    # the last parent is the floor of the midpoint of the array
    # this is the last node with children, so work backwards from it
    last_parent = n // 2 - 1
    for i in range(last_parent, -1, -1):
        min_heapify(nums, n, i)

def min_heapify(nums, heap_size, i):
    """
    Build a min heap of the input number list starting from the ith element
    Ignore any elements after heap-size as heapSort will incrementally
    fill the list with the largest numbers at the back, making the target
    heap smaller and smaller.
    Pretty much same idea as max_heap, but increasing in size instead of descreasing
    """
    # print(f"min_heapify {nums=}, {heap_size=}, {i=}")
    smallest = i  # index of the root node we are considering
    left = 2 * i + 1  # left child index
    right = 2 * i + 2  # right child index

    # if left exists and is larger than the root, it needs to bubble up
    if left < heap_size and nums[left] < nums[smallest]:
        smallest = left

    # similarly if right exists and is larger than the root, it needs to bubble up
    if right < heap_size and nums[right] < nums[smallest]:
        smallest = right

    # if either of the children are larger than the root, swap them out
    # and recursively max_heapify the current tree until no swaps occur
    if smallest != i:
        nums[i], nums[smallest] = nums[smallest], nums[i]  # Swap
        min_heapify(nums, heap_size, smallest)


if __name__ == '__main__':
    unittest.main()
