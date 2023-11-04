"""
Matthew Pacey
CS 514 - HW4 - Greedy Algorithms
"""
import ast
from collections import defaultdict
import heapq
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import time
import unittest

DEBUG = False        # turn on to enable debug prints, False for quiet mode
NUM_RUNS = 20         # increase for more data collection runs

import itertools

def MST_Kruskal(graph):
    """
    Compute the mst for the input graph
    :param graph:
    :return: mst weight and edges
    """
    mst = []                # min span tree
    mst_weight = 0          # weight of all edges in mst
    queue = []              # minheap of all edges in graph, sorted by edge length
    sets = []               # collection of all sets of vertices
    all_vertices = set()    # all vertices in original graph

    # create a min-heap containing all the edges and a set of all vertices
    for start, end, dist in graph:
        heapq.heappush(queue, (dist, (start, end)))
        all_vertices.add(start)
        all_vertices.add(end)

    # start with each vertex in its own set
    for vertex in all_vertices:
        sets.append(set([vertex]))

    # process every edge in the min-heap (breakout condition will check if all vertices are found)
    while queue:
        (dist, (start, end)) = heapq.heappop(queue)  # check shortest edge
        start_set = find_set(sets, start)
        end_set = find_set(sets, end)
        # for the given edge, add it only if the vertices belong to different sets (to avoid cycles)
        if start_set != end_set:
            mst.append((start, end))
            mst_weight += dist
            # combine the old sets into one
            union_set(sets, start_set, end_set)

            if len(sets) == 1 and sets[0] == all_vertices:
                if DEBUG:
                    print("all vertices in mst")
                break

    return mst_weight, mst


def MST_Prim(graph):
    mst = []  # min span tree edges
    mst_weight = 0  # cumulative weight of all edges in mst
    pq = MinQueue() # min priority queue of all unprocessed vertices

    # create adjacency hash of graph for quick lookup later
    adj_hash = defaultdict(dict)
    for start, end, dist in graph:
        adj_hash[start][end] = dist
        adj_hash[end][start] = dist

    # queue up each vertex with an initial distance of infinity
    for vertex in adj_hash.keys():
        pq.add_task(vertex, np.inf)

    # closest vertex of each node in the MST
    parents = {}

    # set the distance of the first vertex to 0 (start of priority queue)
    first = pq.pop_task()
    pq.add_task(first, 0)

    # go until every vertex is processed
    while pq.values:
        u = pq.pop_task()
        neighbors = adj_hash[u].keys()
        for v in neighbors:
            if v not in pq.values:          # only process vertices still in min queue
                continue
            v_dist = adj_hash[u][v]
            pq_weight = pq.values[v]
            if v_dist < pq_weight:
                pq.add_task(v, v_dist)      # update priority in min queue to the closer
                parents[v] = u              # save the closest parent for building MST

    # build the MST from the parent-child vertex hash
    for vertex, parent in parents.items():
        mst.append((parent, vertex))
        mst_weight += adj_hash[vertex][parent]

    return mst_weight, mst


def find_set(sets, vertex):
    """find the set containing the vertex (each vertex can only belong to one)"""
    for subset in sets:
        if vertex in subset:
            return subset
    raise Exception("Vertex not in a known set")


def union_set(sets, start_set, end_set):
    """combine the two sets into one and replace previous two sets with the union set"""
    combined = start_set.union(end_set)
    sets.remove(start_set)
    sets.remove(end_set)
    sets.append(combined)


REMOVED = '<removed-task>'  # placeholder for a removed task
class MinQueue:
    """
    Class to create a min queue with editable priorities
    For our purposes, the 'task' will be the vertex number
    and the priority will be distance to reach that vertex
    Adapted from: https://docs.python.org/3.5/library/heapq.html#priority-queue-implementation-notes
    """
    def __init__(self):
        self.pq = []  # list of entries arranged in a heap
        self.entry_finder = {}  # mapping of tasks to entries
        self.counter = itertools.count()  # unique sequence count
        self.values = {}    # quick lookup of priority

    def add_task(self, task, priority=0):
        """Add a new task or update the priority of an existing task"""
        if task in self.entry_finder:
            self.remove_task(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heapq.heappush(self.pq, entry)
        self.values[task] = priority

    def remove_task(self, task):
        """Mark an existing task as REMOVED.  Raise KeyError if not found."""
        entry = self.entry_finder.pop(task)
        entry[-1] = REMOVED
        del self.values[task]

    def pop_task(self):
        """Remove and return the lowest priority task. Raise KeyError if empty."""
        while self.pq:
            priority, count, task = heapq.heappop(self.pq)
            if task is not REMOVED:
                del self.entry_finder[task]
                del self.values[task]
                return task
        raise KeyError('pop from an empty priority queue')


class Testing(unittest.TestCase):
    def test_kruskal(self):
        graph = ([(0, 1, 1), (0, 2, 5), (1, 2, 1), (2, 3, 2), (1, 3, 6)])
        exp_weight = 4
        exp_edges = [(0, 1), (1, 2), (2, 3)]
        act_weight, act_edges = MST_Kruskal(graph)
        self.assertEqual(exp_weight, act_weight)
        self.assertEqual(set(exp_edges), set(act_edges))

        graph = [(0, 1, 2), (0, 3, 6), (1, 2, 3), (1, 3, 8), (1, 4, 5), (2, 4, 7), (3, 4, 9)]
        exp_weight = 16
        exp_edges = [(0, 1), (0, 3), (1, 2), (1, 4)]
        act_weight, act_edges = MST_Kruskal(graph)
        self.assertEqual(exp_weight, act_weight)
        self.assertEqual(set(exp_edges), set(act_edges))

        graph = [
            (0, 1, 2), (0, 3, 3), (0, 6, 4),
            (1, 2, 3), (1, 4, 2), (3, 4, 5),
            (6, 4, 6), (4, 5, 7)]
        exp_weight = 21
        exp_edges = [(0, 1), (0, 3), (0, 6),
                     (1, 2), (1, 4), (4, 5)]
        act_weight, act_edges = MST_Kruskal(graph)
        self.assertEqual(exp_weight, act_weight)
        self.assertEqual(set(exp_edges), set(act_edges))

    def test_prim(self):
        graph = ([(0,1,1), (0,2,5), (1,2,1), (2,3,2), (1,3,6)])
        exp = (4, [(0, 1), (1, 2), (2, 3)])
        self.assertEqual(exp, MST_Prim(graph))

        graph = [(0, 1, 2), (0, 3, 6), (1, 2, 3), (1, 3, 8), (1, 4, 5), (2, 4, 7), (3, 4, 9)]
        exp_weight = 16
        exp_edges = [(0, 1), (0, 3), (1, 2), (1, 4)]
        act_weight, act_edges = MST_Prim(graph)
        self.assertEqual(exp_weight, act_weight)
        self.assertEqual(set(exp_edges), set(act_edges))

        graph = [
            (0, 1, 2), (0, 3, 3), (0, 6, 4),
            (1, 2, 3), (1, 4, 2), (3, 4, 5),
            (6, 4, 6), (4, 5, 7)]
        exp_weight = 21
        exp_edges = [(0, 1), (0, 3), (0, 6),
                     (1, 2), (1, 4), (4, 5)]
        act_weight, act_edges = MST_Prim(graph)
        self.assertEqual(exp_weight, act_weight)
        self.assertEqual(set(exp_edges), set(act_edges))


def time_single_run(graph, algo):
    """ Return the time in seconds to run <algo> on a graph"""
    start = time.time()
    weight, mst = algo(graph)
    runtime = time.time() - start

    if DEBUG:
        print(f"MST weight with {algo.__name__} = {weight}, {runtime=}")

    return runtime


def data_collection():
    data_root = "test_data"
    test_file_pattern = re.compile(r"generate(\d+)_(\d+)_(\d+).txt")

    test_files = os.listdir(data_root)
    x = [0] * len(test_files)
    nodes = [0] * len(test_files)
    y_prim = [0] * len(test_files)
    y_kruskal = [0] * len(test_files)
    for file in test_files:
        match = test_file_pattern.match(file)
        node_count = int(match.group(1))
        edge_count = int(match.group(2))
        test_num = int(match.group(3)) - 1

        x[test_num] = edge_count
        nodes[test_num] = node_count
        file_path = os.path.join(data_root, file)
        if DEBUG:
            print(f"\nInput file: {file}")
            print(f"{node_count=}, {edge_count=}, {test_num=}")
        with open(file_path, 'r') as fh:
            graph_str = fh.read()
            graph = ast.literal_eval(graph_str)
            for _ in range(NUM_RUNS):
                y_prim[test_num] = time_single_run(graph, MST_Prim) / NUM_RUNS
                y_kruskal[test_num] = time_single_run(graph, MST_Kruskal) / NUM_RUNS

    plt.plot(x, y_prim, c='r', marker='o', label="Prim's Algorithm")
    plt.plot(x, y_kruskal, c='g', marker="o", label="Kruskal")

    plt.title("Greedy Algorithms for Minimum Spanning Tree")
    plt.xlabel("Number of Edges")
    plt.ylabel("Runtime (sec.)")
    plt.legend(loc="lower right")
    plt.savefig("hw5.png")
    plt.show()

    print("Results Table")
    print("Test num, node count, edge count, Prims, Kruskal")
    for i in range(len(x)):
        print(f"{i+1}, {nodes[i]}, {x[i]}, {y_prim[i]:1.5f}, {y_kruskal[i]:1.5f}")

    print("Pri")
    for i in range(len(x)):
        print(f"{y_prim[i]:1.5f}")

    print("Kru")
    for i in range(len(x)):
        print(f"{y_kruskal[i]:1.5f}")

if __name__ == '__main__':
    data_collection()
    if DEBUG:
        unittest.main()
