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
    if len(nums) > 1:
        mid = len(nums) // 2
        left = merge_sort(nums[:mid])
        right = merge_sort(nums[mid:])

        left_i, right_i, num_i = 0, 0, 0
        while left_i < len(left) and right_i < len(right):
            if left[left_i] < right[right_i]:
                nums[num_i] = left[left_i]
                left_i += 1
            else:
                nums[num_i] = right[right_i]
                right_i += 1
            num_i += 1


        while left_i < len(left):
            nums[num_i] = left[left_i]
            left_i += 1
            num_i += 1
        while right_i < len(right):
            nums[num_i] = right[right_i]
            right_i += 1
            num_i += 1


    return nums


def quick_sort_delete(nums):
    # pick a pivot element in the list
    # divide list into 2 sublists:
    #   left: elements < pivot
    #   right: elements >= pivot
    # recursively sort the 2 sublists
    # concatenate sublists with pivot in the middle

    if nums == []:
        return []
    else:
        #pivot_num = 0
        pivot_num = random.choice(range(len(nums)))
        pivot = nums[pivot_num]
        left = [x for x in nums[pivot_num] if x < pivot]
        right = [x for x in nums[pivot_num + 1:] if x >= pivot]

        return quick_sort_all(left) + [pivot] + quick_sort_all(right)


def quick_sort_all(nums):
    return quick_sort(nums, 0, len(nums) - 1)


def quick_sort(nums, left, right):
    if left < right:
        pivot_i = partition(nums, left, right)
        quick_sort(nums, left, pivot_i - 1)
        quick_sort(nums, pivot_i + 1, right)
    return nums


def partition(nums, left, right):
    # TODO: try random num
    pivot = nums[right]
    pivot = nums[left]
    pivot = nums[left + (right - left) // 2]
    # pivot = nums[random.randint(left, right - 1)]
    print(f"{left=}, {right=}, {pivot=}")
    i = left - 1

    for j in range(left, right):
        if nums[j] <= pivot:
            i = i + 1
            nums[i], nums[j] = nums[j], nums[i]

    nums[i + 1], nums[right] = nums[right], nums[i + 1]

    return i + 1
class Testing(unittest.TestCase):
    def test_small(self):
        # sort ordered list
        nums = list(range(0, 10))
        exp = sorted(nums)
        self.assertEqual(merge_sort(nums), exp)
        self.assertEqual(quick_sort_all(nums), exp)

        # sort reverse ordered list
        nums = list(range(10, 0))
        exp = sorted(nums)
        self.assertEqual(merge_sort(nums), exp)
        self.assertEqual(quick_sort_all(nums), exp)

        # sort randomly ordered list
        nums = list(range(10, 0))
        random.shuffle(nums)
        exp = sorted(nums)
        self.assertEqual(merge_sort(nums), exp)
        self.assertEqual(quick_sort_all(nums), exp)


def time_single_run(nums, algo):
    """ Return the time in seconds to run <algo>sort on a list of nums"""
    start = time.time()
    print(f"Sorting ({nums}", end="\r")
    sort = algo(nums)
    runtime = time.time() - start

    # error checking
    print(f"{algo=}")
    print(f"{nums=}")
    print(f"{sort=}")
    assert(sort == sorted(nums))
    return runtime

def data_collection():

    nums = list(range(0, 200))
    num_runs = 20

    merge_sum_sorted = 0
    quick_sum_sorted = 0
    merge_sum_shuffled = 0
    quick_sum_shuffled = 0

    for _ in range(num_runs):
        quick_sum_sorted += time_single_run(nums, quick_sort_all)
        merge_sum_sorted += time_single_run(nums, merge_sort)

    for _ in range(num_runs):
        random.shuffle(nums)
        quick_sum_shuffled += time_single_run(nums, quick_sort_all)
        merge_sum_shuffled += time_single_run(nums, merge_sort)

    print("sorted")
    print(f"merge {merge_sum_sorted / num_runs}")
    print(f"quick {quick_sum_sorted / num_runs}")

    print("shuffled")
    print(f"merge {merge_sum_shuffled / num_runs}")
    print(f"quick {quick_sum_shuffled / num_runs}")

def data_collection_old():

    i = 1
    MAX_I = 3
    num_runs = 20
    base = pow(10, i - 1)

    # for plotting
    x, y = [], []

    while i <= MAX_I:
        # for each 'i' digit number, produce j random numbers and factor them
        sum = 0
        for j in range(0, num_runs):
            rand = random.randint(base, base * 10)
            runtime = time_single_run(rand)
            sum += runtime
            print('%d :: %f sec' % (rand, runtime), end='\r')
            # log to file (opening handle each loop in case runtime is really long, won't lose all data)
            with open('runtimes_large_n.csv', 'a') as out_fh:
                out_fh.write('%d,%f\n' % (rand, runtime))

        average = sum / num_runs
        print('Average (%d): %f' % (i, average))
        x.append(i)
        y.append(average)
        with open('runtime_averages.csv', 'a') as out_fh:
            out_fh.write('%i,%f\n' % (i, average))
        # increment the base random number by a digit
        base = base * 10
        i += 1

    # calculate equation for quadratic trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    z4 = np.polyfit(x, y, 5)
    p4 = np.poly1d(z4)
    # add trend line to plot
    plt.plot(x, p(x), marker="x", c="red", label="Linear Runtime Trendline")
    plt.plot(x, p4(x), marker="x", c="green", label="$n^4$ Runtime Trendline", alpha=0.5)

    plt.xticks(x)
    plt.scatter(x, y, marker="o", label="Avg. Runtime", s=100)


    print("Linear Trend equation: y=%.6fx+(%.6f)" % (z[0], z[1]))
    print("n^4 Trend equation: y=%.6fx^4 + %.6fx^3 + %.6fx^2 + %.6fx+(%.6f)" % (z4[0], z4[1], z4[2], z4[3], z4[4]))

    plt.title("Factors Runtime Experiments")
    plt.xlabel("Number of input digits")
    plt.ylabel(f"Average runtime of {num_runs} trials (sec)")
    plt.legend(loc="upper left")
    plt.savefig("runtimes2.png")
    plt.show()


if __name__ == '__main__':
    unittest.main()
    # data_collection()
