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
# because our loop starts at
known_primes = [2, 3]

def is_prime(num):
    """
    Return True iff input number is prime, else False
    :param num:
    :return:
    """
    global known_primes
    # error checking
    if num <= 1:
        return False
    # check if we've seen it/marked it as a prime before, constant lookup
    if num in known_primes:
        return True

    # a number is prime if no other numbers can be multiplied to make the number
    # time complexity: modulo operation of each known prime
    # assuming empty cache, this has no performance impact O(1)
    for prime in known_primes:
        if (num % prime) == 0:
            return False

    # if no numbers can divide without a remainder (modulo 0 result), the number is prime
    # we already checked evens, so only need to check all odd numbers
    # we can start at 3 and increment counter by 2 to skip all evens, square root
    # worst case scenario, the for loop iterates for every other number from 3 to sqrt of num
    # time complexity:       so O(sqrt(n)/2) -> O(sqrt(n))
    stop_val = math.ceil(math.sqrt(num)) + 1
    for i in range(5, stop_val, 2):
        if (num % i) == 0:
            return False

    # if here, no divisor has been found, the number is prime
    known_primes.append(num)       # cache for later
    return True

    # final time complexity (for all of is_prime): O(sqrt(n)/2)


def factors(num):
    """
    Return the prime factors for the given number
    If the number is prime, return empty list
    If the number is not prime, return the smallest prime numbers that
        multiply to get the number
        e.g.
            factors(5) = [] # 5 is prime
            factors(12) = [2, 2, 3] # not prime
    :param num:
    :return:
    """
    # cache for faster subsequent runs
    global known_primes
    if num <= 1:
        return []

    # complexity: O(sqrt(n)/2)
    if is_prime(num):
        return []

    # start counting primes from the smallest, add them to list
    primes = []

    # first remove the known primes
    # if the cache is not used, this will have no impact on performance
    # if the cache contains all primes that make up the input, the algo
    # will effectively terminate here; O(n) where n is the size of the cache
    for prime in known_primes:
        while (num % prime) == 0:
            primes.append(prime)
            num = num // prime

    # next find the smallest factor and restart with num divided by smallest factor, checking only odd numbers
    # since we are counting up, we only need to check up to the square root of the number
    # because sqrt(n) * sqrt(n) + 1 >= n (no new factors will be discovered past here)
    # complexity: O(sqrt(n)/2)
    stop_val = math.ceil(math.sqrt(num)) + 1
    # O(sqrt(n)/2)
    for i in range(5, stop_val, 2):
        if (num % i) == 0:
            # O(sqrt(n)/2)
            if is_prime(i):
                primes.append(i)
            new_num = num // i  # complexity: O(1); single divide operation

            sub_factors = factors(new_num)
            # if no sub factors are found, the new_num is a prime
            if not sub_factors:
                sub_factors = [new_num]
            return primes + sub_factors

    # return the prime factors collected
    return primes


def factors_slow(input_num):
    """
    Single algo but without caching it is much slower
    :param input_num:
    :return:
    """
    num = input_num
    if num <= 3:
        return []

    primes = []
    # remove all evens from the number,
    while (num % 2) == 0:
        num = num // 2
        primes.append(2)

    # complexity: O(sqrt(n)/2)
    stop_val = math.ceil(math.sqrt(num)) + 1
    for i in range(3, stop_val, 2):
        while (num % i) == 0:
            sub_factors = factors_slow(i)
            if sub_factors:
                primes += sub_factors
            else:
                primes.append(i)
            num = num // i

    if num > 1 and input_num != num:
        primes.append(num)

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
        self.assertEqual(factors(8889689999934465613), [43, 42083, 4912600735277])
        self.assertEqual(factors(80497520654571648085), [5, 21221, 2769593, 273924389])


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
    i = 15
    MAX_I = 22
    num_runs = 5
    base = pow(10, i - 1)

    # for plotting
    x, y = [], []
    # x, y = [17, 18, 19, 20], [0.357950, 1.226209, 54.058740, 84.985157]
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

    plt.scatter(x, y, marker="o", label="Avg. Runtime")
    # calculate equation for quadratic trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    z4 = np.polyfit(x, y, 5)
    p4 = np.poly1d(z4)
    # add trend line to plot
    plt.plot(x, p(x), marker="x", c="red", label="Linear Runtime Trendline")
    plt.plot(x, p4(x), marker="x", c="green", label="$n^4$ Runtime Trendline")

    plt.xticks(x)

    print("Linear Trend equation: y=%.6fx+(%.6f)" % (z[0], z[1]))
    print("n^4 Trend equation: y=%.6fx^4 + %.6fx^3 + %.6fx^2 + %.6fx+(%.6f)" % (z4[0], z4[1], z4[2], z4[3], z4[4]))

    plt.title("Factors Runtime Experiments")
    plt.xlabel("Number of input digits")
    plt.ylabel(f"Average runtime of {num_runs} trials (sec)")
    plt.legend(loc="upper left")
    plt.savefig("runtimes.png")
    plt.show()


if __name__ == '__main__':
    # unittest.main()
    data_collection()
