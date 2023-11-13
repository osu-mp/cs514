"""
Matthew Pacey
CS 514 - HW6 - Dynamic Programming
"""
import string
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import unittest

DEBUG = False        # turn on to enable debug prints, False for quiet mode
NUM_RUNS = 10         # increase for more data collection runs


def editDistance(str1, str2):
    """
    The basic edit distance algorithm is based on dynamic programming and has the complexity of O(mn).
    There are 3 possible operations, insert (I),  replace (R), and delete (D), each of which
    costs 1 unit (unless the characters exactly match).
    For example, if x= [babble] and y='apple', the edit distance is 3 (delete b, replace b/p):
        babble
         apple
    :param str1:
    :param str2:
    :return:
    """
    m = len(str1)
    n = len(str2)

    # create distance matrix
    dist_matr = np.zeros((m + 1, n + 1), dtype=int)

    # initialize first row and col to compare each string to empty other string
    # this mean the edit distance will be the length of the non-empty string up to that point
    for i in range(m + 1):
        dist_matr[i][0] = i

    for j in range(n + 1):
        dist_matr[0][j] = j

    # next iterate over the non-empty strings (starting at index 1) and compute the cost
    # for inserting, deleting, and substituing a character
    # save the value with the lowest cost
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            insert = dist_matr[i][j - 1] + 1
            delete = dist_matr[i - 1][j] + 1
            substitution = dist_matr[i - 1][j - 1]
            # substitution only incurs a cost if the characters differ
            if str1[i - 1] != str2[j - 1]:
                substitution += 1

            dist_matr[i][j] = min(insert, delete, substitution)

    # final edit distance is the bottom right value of the matrix
    return dist_matr[m][n]


class Testing(unittest.TestCase):
    def test_simple(self):
        act = editDistance("A", "A")
        self.assertEqual(0, act)

        act = editDistance("ABC", "BCD")
        self.assertEqual(2, act)

        act = editDistance("A", "B")
        self.assertEqual(1, act)

        act = editDistance("ABCDEFGH", "AACRFHI")
        self.assertEqual(5, act)

    def test_given(self):
        act = editDistance("babble", "apple")
        self.assertEqual(3, act)
        act = editDistance("ATCAT", "ATTATC")
        self.assertEqual(2, act)
        act = editDistance(
            "taacttctagtacatacccgggttgagcccccatttcttggttggatgcgaggaacattacgctagaggaacaacaaggtcagaggcctgttactcctat",
            "taacttctagtacatacccgggttgagcccccatttccgaggaacattacgctagaggaacaacaaggtcagaggcctgttactcctat")
        self.assertEqual(11, act)
        act = editDistance("CGCAATTCTGAAGCGCTGGGGAAGACGGGT", "TATCCCATCGAACGCCTATTCTAGGAT")
        self.assertEqual(18, act)
        act = editDistance(
            "tatttacccaccacttctcccgttctcgaatcaggaatagactactgcaatcgacgtagggataggaaactccccgagtttccacagaccgcgcgcgatattgctcgccggcatacagcccttgcgggaaatcggcaaccagttgagtagttcattggcttaagacgctttaagtacttaggatggtcgcgtcgtgccaa",
            "atggtctccccgcaagataccctaattccttcactctctcacctagagcaccttaacgtgaaagatggctttaggatggcatagctatgccgtggtgctatgagatcaaacaccgctttctttttagaacgggtcctaatacgacgtgccgtgcacagcattgtaataacactggacgacgcgggctcggttagtaagtt")
        self.assertEqual(112, act)


def time_single_run(str1, str2):
    """ Return the time in seconds to editDistance on the given input"""
    start = time.time()
    dist = editDistance(str1, str2)
    runtime = time.time() - start

    if DEBUG:
        print(f"{len(str1)=}, {dist=}, {runtime=}")

    return runtime


def random_str(num_chars):
    # generate a random string of length num_chars
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for _ in range(num_chars))


def data_collection():
    x = []
    y = []
    for i in range(1, 11):
        num_chars = 100 * i
        x.append(num_chars)

        str1 = random_str(num_chars)
        str2 = random_str(num_chars)

        runtime = 0
        for _ in range(NUM_RUNS):
            runtime += time_single_run(str1, str2) / NUM_RUNS

        y.append(runtime)


    plt.plot(x, y, c='r', marker='o')
    plt.title("Runtime of editDistance with Random Strings")
    plt.xlabel("Number of Characters in Input Strings")
    plt.ylabel("Runtime (sec.)")
    plt.savefig("hw6.png")
    plt.show()

    print("Results Table")
    print("Test num, num chars, runtime")
    for i in range(len(x)):
        print(f"{i+1}, {x[i]}, {y[i]:1.5f}")


def longest_common_substring(str1, str2):
    m = len(str1)
    n = len(str2)

    # initialize a table to store the longest
    lcs_matrix = np.zeros((m + 1, n + 1), dtype=int)

    max_len = 0
    max_pos = None

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                lcs_matrix[i][j] = lcs_matrix[i - 1][j - 1] + 1
                if lcs_matrix[i][j] > max_len:
                    max_len = lcs_matrix[i][j]
                    max_pos = (i, j)
            else:
                lcs_matrix[i][j] = 0

    if max_len == 0:
        return None

    # these should match, just checking
    longest_common_i = str1[max_pos[0] - max_len:max_pos[0]]
    longest_common_j = str1[max_pos[1] - max_len:max_pos[1]]
    assert(longest_common_j == longest_common_j)
    return longest_common_i


def coffee_shops_greedy(shops, k):
    selected_shops = set()
    available_shops = set(shops.keys())
    total_profit = 0

    while available_shops:
        # assuming each potential store has a positive profit
        max_profit, max_name = -1, None

        for shop_name in available_shops:
            if shops[shop_name]["profit"] > max_profit:
                max_name = shop_name
                max_profit = shops[shop_name]["profit"]

        if max_name:
            selected_shops.add(max_name)
            available_shops.remove(max_name)
            total_profit += max_profit

            for adj_name in list(available_shops):
                if abs(shops[max_name]['dist'] - shops[adj_name]['dist']) < k:
                    available_shops.remove(adj_name)

    return total_profit


def coffee_shops_dynamic(shops, k):
    shop_count = len(shops)
    memo = {}

    shop_list = list(shops.values())

    def check_location(i):
        if i < 0:
            return 0
        if i in memo:
            return memo[i]

        skip_current = check_location(i - 1)
        select_current = shop_list[i]['profit'] + check_location(find_previous_valid_location(i, k))

        max_profit_at_i = max(skip_current, select_current)

        memo[i] = max_profit_at_i
        return max_profit_at_i

    def find_previous_valid_location(i, k):
        for j in range(i - 1, -1, -1):
            if shop_list[i]['dist'] - shop_list[j]['dist'] > k:
                return j

        return -1

    max_profit = check_location(shop_count - 1)

    return max_profit


def coffee_shops_test():
    shops = {
        'A': {'dist': 5, 'profit': 10},
        'B': {'dist': 10, 'profit': 11},
        'C': {'dist': 15, 'profit': 10},
    }
    k = 6
    print(f"Greedy {coffee_shops_greedy(shops, k)}")
    assert(coffee_shops_greedy(shops, k) == 11)
    print(f"Dynamic {coffee_shops_dynamic(shops, k)}")
    assert(coffee_shops_dynamic(shops, k) == 20)


if __name__ == '__main__':
    data_collection()
    if DEBUG:
        assert("anthropi" == longest_common_substring("Philanthropic", "Misanthropist"))
        assert(None == longest_common_substring("ABC", "DEF"))
        assert ("BC" == longest_common_substring("ABC", "BCDEFG"))
        coffee_shops_test()
        unittest.main()
