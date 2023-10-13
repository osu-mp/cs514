"""
Matthew Pacey
CS 514 - HW2 - Divide and Conquer Sorting
"""
import math

import numpy as np
import matplotlib.pyplot as plt
import random
import time
import unittest


def merge_sort(nums):
    """
    Merge sort:
    -divide input list into two equal parts
    -sort each individual list
    -combine both lists by iterating over each and add the smaller value to the final list
    """

    if len(nums) > 1:
        mid = len(nums) // 2                # split list at the midpoint
        left = merge_sort(nums[:mid])       # sort each individually
        right = merge_sort(nums[mid:])

        # with left and right sorted, iterate over both lists
        # and add the smaller item to the final list
        left_i, right_i, num_i = 0, 0, 0
        while left_i < len(left) and right_i < len(right):
            if left[left_i] < right[right_i]:
                nums[num_i] = left[left_i]
                left_i += 1
            else:
                nums[num_i] = right[right_i]
                right_i += 1
            num_i += 1

        # at this point one ore more of the lists is empty, so we need to empty
        # any remaining values from the lists (will either be right, left, or neither with values)
        while left_i < len(left):
            nums[num_i] = left[left_i]
            left_i += 1
            num_i += 1
        while right_i < len(right):
            nums[num_i] = right[right_i]
            right_i += 1
            num_i += 1

    # else: base case when 0 or 1 items are in the list (sorted)

    return nums



def quick_sort(nums, random_pivot=False):
    """
    quick sort:
    -pick a pivot point in the list (using either midpoint or random index, default midpoint)
    -put all items less than pivot in left list
    -put all items greater than pivot in right list
    -put all items matching pivot in mid list
    -recursively quicksort the right and left lists until there are 0 or 1 items in the list
    -combine the left, mid, and right lists (all sorted by this point)
    """

    # base case: 0 or 1 items in a list is sorted
    if len(nums) <= 1:
        return nums
    else:
        # select either a random pivot point or the midpoint of the list
        # using first or last index leads to too many recursive calls
        if random_pivot:
            pivot_num = random.choice(range(len(nums)))
        else:
            pivot_num = len(nums) // 2
        # get the value of the list at that pivot num
        pivot = nums[pivot_num]
        # divide the larger list into three parts: less than, equal to, greater than
        left, mid, right = [], [], []
        for num in nums:
            if num < pivot:
                left.append(num)
            elif num > pivot:
                right.append(num)
            else:
                mid.append(num)

        # recursively sort the left/right lists and combine with pivot matches/mid
        return quick_sort(left) + mid + quick_sort(right)


def quick_sort_random(nums):
    """
    Run same quick sort algo but pick a random pivot instead of midpoint
    """
    return quick_sort(nums, random_pivot=True)


class Testing(unittest.TestCase):
    def test_small(self):
        # sort ordered list
        nums = list(range(0, 10))
        exp = sorted(nums)
        self.assertEqual(merge_sort(nums), exp)
        self.assertEqual(quick_sort(nums), exp)
        self.assertEqual(quick_sort(nums, random_pivot=True), exp)

        # sort reverse ordered list
        nums = list(range(10, 0))
        exp = sorted(nums)
        self.assertEqual(merge_sort(nums), exp)
        self.assertEqual(quick_sort(nums), exp)

        # sort randomly ordered list
        nums = list(range(10, 0, -1))
        random.shuffle(nums)
        exp = sorted(nums)
        self.assertEqual(merge_sort(nums), exp)
        self.assertEqual(quick_sort(nums), exp)

        # sort randomly ordered list with duplicates
        nums = list(range(10, 0, -1)) + list(range(10, 0, -1))
        random.shuffle(nums)
        exp = sorted(nums)
        self.assertEqual(merge_sort(nums), exp)
        self.assertEqual(quick_sort(nums), exp)

    def test_large(self):
        nums = list(range(0, 100000))
        exp = sorted(nums)
        self.assertEqual(merge_sort(nums), exp)
        self.assertEqual(quick_sort(nums), exp)
        self.assertEqual(quick_sort(nums, random_pivot=True), exp)

        random.shuffle(nums)
        self.assertEqual(merge_sort(nums), exp)
        self.assertEqual(quick_sort(nums), exp)
        self.assertEqual(quick_sort(nums, random_pivot=True), exp)


def time_single_run(nums, algo):
    """ Return the time in seconds to run <algo>sort on a list of nums"""
    start = time.time()
    # print(f"Sorted {len(nums)} items with {algo}")
    sort = algo(nums)
    runtime = time.time() - start

    # error check to ensure that the list was acutally sorted (compare with built-in sort)
    # assert(sort == sorted(nums))
    return runtime


def data_collection():

    y_merge_sorted = []
    y_quick_mid_sorted = []
    y_quick_rand_sorted = []
    y_merge_shuffled = []
    y_quick_mid_shuffled = []
    y_quick_rand_shuffled = []
    x = []
    # x_ticks = []

    # for i in range(3, 8):
    for max_num in [pow(10, 5), 3 * pow(10, 5), 6 * pow(10, 5),
                    pow(10, 6), 3 * pow(10, 6), 6 * pow(10, 6),
                    pow(10, 7)]:
        print(f"Results for list of size {max_num}")
        # max_num = pow(10, i)
        nums = list(range(0, max_num))
        num_runs = 10
        x.append(max_num)
        # x_ticks.append(max_num)

        merge_sum_sorted = 0
        quick_mid_sum_sorted = 0
        quick_rand_sum_sorted = 0
        merge_sum_shuffled = 0
        quick_mid_sum_shuffled = 0
        quick_rand_sum_shuffled = 0

        for _ in range(num_runs):
            merge_sum_sorted += time_single_run(nums, merge_sort)
            quick_mid_sum_sorted += time_single_run(nums, quick_sort)
            # quick_rand_sum_sorted += time_single_run(nums, quick_sort_random)

        for _ in range(num_runs):
            random.shuffle(nums)
            merge_sum_shuffled += time_single_run(nums, merge_sort)
            quick_mid_sum_shuffled += time_single_run(nums, quick_sort)
            # quick_rand_sum_shuffled += time_single_run(nums, quick_sort_random)

        y_merge_sorted.append(merge_sum_sorted / num_runs)
        y_quick_mid_sorted.append(quick_mid_sum_sorted / num_runs)
        # y_quick_rand_sorted.append(quick_rand_sum_sorted / num_runs)
        y_merge_shuffled.append(merge_sum_shuffled / num_runs)
        y_quick_mid_shuffled.append(quick_mid_sum_shuffled / num_runs)
        # y_quick_rand_shuffled.append(quick_rand_sum_shuffled / num_runs)

        print("sorted")
        print(f"merge {merge_sum_sorted / num_runs}")
        print(f"quick (mid) {quick_mid_sum_sorted / num_runs}")
        # print(f"quick (rand) {quick_rand_sum_sorted / num_runs}")

        print("shuffled")
        print(f"merge {merge_sum_shuffled / num_runs}")
        print(f"quick (mid) {quick_mid_sum_shuffled / num_runs}")
        # print(f"quick (rand) {quick_rand_sum_shuffled / num_runs}")


    plt.plot(x, y_merge_sorted, c='r', marker='o', label="Merge Sort (sorted input)")
    plt.plot(x, y_merge_shuffled, c='g', marker="o", label="Merge Sort (shuffled input)")

    plt.plot(x, y_quick_mid_sorted, c='b', marker='o', label="Quick Sort (sorted input)")
    plt.plot(x, y_quick_mid_shuffled, c='black', marker="o", label="Quick Sort (shuffle input)")

    # plt.plot(x, y_quick_rand_sorted, c='g', label="Quick Sort (random pivot)")

    plt.title("Sorting Comparison with Sorted and Unsorted Inputs")
    plt.xlabel("Number of input digits")
    plt.ylabel(f"Average runtime of {num_runs} trials (sec)")

    # plt.plot(x, y_quick_rand_shuffled, c='g', label="Quick Sort (random pivot)")
    # plt.xticks(x_ticks)

    plt.legend(loc="upper left")
    plt.savefig("combined.png")
    plt.show()

if __name__ == '__main__':
    # unittest.main()
    data_collection()
