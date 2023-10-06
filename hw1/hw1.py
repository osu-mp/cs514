"""
Matthew Pacey
CS 514 - HW1 - Prime Factors
"""
import math

import numpy as np
import matplotlib.pyplot as plt
import random
import time
import unittest

# keep a cache of already calculated primes to speed up subsequent runs
known_primes = [2, 3]
max_loop = 3

def is_prime(num):
    global known_primes, max_loop

    if num <= 1:
        return False
    if num in known_primes:
        return True

    # a number is prime if no other numbers can be multiplied to make the number
    # base case: if we've seen it/marked it as a prime before
    # time complexity: O(1), constant time; single modulo operation
    for prime in known_primes:
        if (num % prime) == 0:
            return False

    # if no numbers can divide without a remainder (modulo 0 result), the number is prime
    # we already checked evens, so only need to check all odd numbers
    # we can start at 3 or previous high loop value and increment counter by 2 to skip all evens, only going to half of num
    # time complexity: O(n)
    #       worst case scenario, the for loop iterates for every other number from 3 to half of num
    #       roughly n/2 iterations, so it is O(n)
    stop_val = math.ceil(math.sqrt(num))
    for i in range(max_loop, stop_val, 2):
        if (num % i) == 0:
            max_loop = i
            return False

    # if here, no divisor has been found, the number is prime
    known_primes.append(num)       # cache for later
    return True

    # time complexity (for all of is_prime): O(n)

def factors(num):
    global max_loop, known_primes
    if num <= 1:
        return []

    if is_prime(num):
        return []

    # start counting primes from the smallest, add them to list
    primes = []
    # first remove the known primes
    for prime in known_primes:
        while (num % prime) == 0:
            primes.append(prime)
            num = num // prime

    # next find the smallest factor and restart with num divided by smallest factor, checking only odd numbers
    stop_val = math.ceil(math.sqrt(num))
    for i in range(max_loop, stop_val, 2):  # complexity: O(n); worst case it iterates n/2 times
        max_loop = i
        if (num % i) == 0:  # complexity: O(1); single modulo operation
            if is_prime(i):
                primes.append(i)
            new_num = num // i  # complexity: O(1); single divide operation
            # complexity (below): O(n^2)
            #   worst case this iterates n times (above for loop), and executes O(n) is_prime via factors func
            #   O(n * n) -> O(n^2)
            sub_factors = factors(new_num)
            # if no subfactors are found, the new_num is a prime
            if not sub_factors:
                sub_factors = [new_num]
            return primes + sub_factors

    # return the prime factors collected
    return primes


class Testing(unittest.TestCase):
    def test_invalid_inputs(self):
        self.assertEqual(factors(-100), [])
        self.assertEqual(factors(0), [])
        self.assertEqual(factors(1), [])

    def test_small(self):
        self.assertEqual(factors(2), [])
        self.assertEqual(factors(3), [])
        self.assertEqual(factors(4), [2, 2])
        self.assertEqual(factors(5), [])
        self.assertEqual(factors(6), [2, 3])
        self.assertEqual(factors(7), [])
        self.assertEqual(factors(8), [2, 2, 2])
        self.assertEqual(factors(9), [3, 3])
        self.assertEqual(factors(10), [2, 5])
        self.assertEqual(factors(11), [])
        self.assertEqual(factors(12), [2, 2, 3])
        self.assertEqual(factors(27), [3, 3, 3])
        self.assertEqual(factors(36), [2, 2, 3, 3])

    def test_random_large(self):
        self.assertEqual(factors(23426), [2, 13, 17, 53])
        self.assertEqual(factors(600851475143), [71, 839, 1471, 6857])

    def test_large_prime(self):
        # each is a large prime, should return empty list
        self.assertEqual(factors(1266008519), [])
        # subsequent runs should run in 0(1) time because of cache
        self.assertEqual(factors(1266008519), [])
        self.assertEqual(factors(1266008519), [])


def time_single_run(num):
    """ Return the time in seconds to run factors on num"""
    start = time.time()
    print(f"Running factors({num}", end="\r")
    factors(num)
    runtime = time.time() - start
    return runtime

def data_collection():

    # i is the number of digits in our input number (the random nums for that loop will be between 10^i-1 and 10^i)
    # the range is huge, but there's a user input check to break out of the loop when desired
    i = 8
    MAX_I = 22
    num_runs = 100
    base = pow(10, i - 1)

    # for plotting
    x, y = [], []
    # while True:
    while i <= MAX_I:
        # for each 'i' digit number, produce j random numbers and factor them
        sum = 0
        for j in range(0, num_runs):
            rand = random.randint(base, base * 10)
            runtime = time_single_run(rand)
            sum += runtime
            print('%d :: %f sec' % (rand, runtime))
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

        # response = input('Increase one more digit? (y or n)')
        # if response.lower() != 'y':
        #     print('Quitting')
        #     exit(1)

    plt.scatter(x, y, marker="o")
    # calculate equation for quadratic trendline
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    # add trendline to plot
    plt.plot(x, p(x), marker="x", c="red")

    print("y=%.6fx^2+%.6fx+(%.6f)" % (z[0], z[1], z[2]))

    plt.xlabel("Number of input digits")
    plt.ylabel(f"Average runtime of {num_runs} trials (sec)")
    plt.title("Factors Runtime Experiments")
    plt.savefig("runtimes.png")
    plt.show()

if __name__ == '__main__':
    # unittest.main()
    data_collection()