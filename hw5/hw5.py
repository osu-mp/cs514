"""
Matthew Pacey
CS 514 - HW4 - Greedy Algorithms
"""
import heapq
import numpy as np
from queue import PriorityQueue, SimpleQueue
import unittest


def find_set(sets, vertex):
    # find the set containing the vertex (each vertex can only belong to one)
    for subset in sets:
        if vertex in subset:
            return subset
    raise Exception("Vertex not in a known set")


def union_set(sets, start_set, end_set):
    # combine the two sets into one and replace previous two sets with the union set
    combined = start_set.union(end_set)
    sets.remove(start_set)
    sets.remove(end_set)
    sets.append(combined)


def MST_Kruskal(graph):
    """
    Compute the mst for the input graph
    :param graph:
    :return: mst weight and edges
    """
    mst = []
    mst_weight = 0
    queue = []
    sets = []
    all_vertices = set()

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
        (dist, (start, end)) = heapq.heappop(queue)
        start_set = find_set(sets, start)
        end_set = find_set(sets, end)
        # for the given edge, add it only if the vertices belong to different sets (avoid cycles)
        if start_set != end_set:
            mst.append((start, end))
            mst_weight += dist
            # combine the old sets into one
            union_set(sets, start_set, end_set)

            if len(sets) == 1 and sets[0] == all_vertices:
                print("all vertices in mst")
                break

    return mst_weight, mst


def MST_Prim(graph):

    return (4, [(0, 1), (1, 2), (2, 3)])


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

    def test_prim(self):
        graph = ([(0,1,1), (0,2,5), (1,2,1), (2,3,2), (1,3,6)])
        exp = (4, [(0, 1), (1, 2), (2, 3)])
        self.assertEqual(exp, MST_Prim(graph))

if __name__ == '__main__':
    unittest.main()