"""
Matthew Pacey
CS 514 - HW8 - Linear Programming and Network Flow
"""

from collections import defaultdict, deque
import unittest

DEBUG = False        # turn on to enable debug prints, False for quiet mode


def Max_Flow_Fat(input):
    """
    In this approach, the augmented flows are computed by finding a path from
     source to sink (an st-path) in the residual graph that has a maximum capacity.
     The capacity of the path is the minimum (remaining) capacity of any
     edge (the 'bottleneck') in that path. The algorithm works like Dijkstra's
     algorithm except that it tries to build a maximum capacity path (rather than
     the shortest path) from the source to sink.
    :param source:
    :param sink:
    :param edges:
    :return:
    """
    source, sink, edges = input
    if DEBUG:
        print(f"Fat flow: {source=}, {sink=}")

    graph = defaultdict(list)
    for u, v, capacity in edges:
        graph[u].append((v, capacity))
        graph[v].append((u, 0))  # add reverse edges with 0 capacity

    flow_parents = {}  # for the given path, key=node and value=node's parent
    max_flow = 0       # max flow for all discovered paths

    # all paths added during augmentation; key=parent, value: dictionary of node/capacity for each child of parent
    all_paths = defaultdict(lambda: defaultdict(int))
    while dfs(graph, source, sink, flow_parents):
        path_flow = float("inf")  # maximum flow for current path (constrained by min capacity of each pipe)
        s = sink                    # work backwards from sink towards source
        path = {}

        # constrain max flow as lowest capacity in the augmenting path
        while s != source:
            parent_node = flow_parents[s]
            for u, capacity in graph[parent_node]:
                if u == s:
                    path_flow = min(path_flow, capacity)
                    path[parent_node] = s
                    break
            s = flow_parents[s]

        # Update residual capacities and reverse edges along the path
        v = sink
        while v != source:
            parent_node = flow_parents[v]
            for u, capacity in graph[parent_node]:
                if u == v:
                    parent_node = flow_parents[v]
                    # remove previous capacity and add with updated value
                    graph[parent_node].remove((v, capacity))
                    graph[parent_node].append((v, capacity - path_flow))
                    graph[v].append((u, capacity + path_flow))
                    break
            v = flow_parents[v]

        # save the discovered path to be returned at end
        for node, child in path.items():
            all_paths[node][child] += path_flow
            # print(f"adding path flow ({path_flow}) to all paths {node=}, {child=}")

        max_flow += path_flow
        flow_parents = {}  # clear parents for next loop

    path_list = []
    for node in sorted(all_paths.keys()):
        for child in sorted(all_paths[node].keys()):
            path_list.append((node, child, all_paths[node][child]))
            if DEBUG:
                print(f"path from {node}->{child}, {capacity=}")

    if DEBUG:
        print(f"{max_flow=}\n")
    return max_flow, path_list


def dfs(graph, source, sink, flow_parents):
    """
    Depth first search to find a path from source to sink with available capacity
    :param graph:
    :param source:
    :param sink:
    :param flow_parents: Dictionary where key is child node, value is parent of that child
    :return: True if some path with capacity exists, else False
    """
    visited = set()
    queue = deque()
    queue.append(source)
    visited.add(source)

    while queue:
        parent = queue.popleft()
        for child, capacity in graph[parent]:
            if child not in visited and capacity > 0:
                queue.append(child)
                visited.add(child)
                flow_parents[child] = parent
    return sink in visited


def Max_Flow_Short(input):
    """
    A breadth first search in the residual graph which returns the shortest path
     (i.e., the path with the least number of edges) with non-zero flow.
    :param source:
    :param sink:
    :param edges:
    :return:
    """
    source, sink, edges = input
    if DEBUG:
        print(f"Short Pipes: {source=}, {sink=}")

    graph = defaultdict(list)
    for u, v, capacity in edges:
        graph[u].append((v, capacity))  # forward: capacity of each edge
        graph[v].append((u, 0))  # reverse: residual

    flow_parents = {}       # for the given path, key=node and value=node's parent
    max_flow = 0            # cumulative flow, increases with each augmentation

    # all paths added during augmentation; key=parent, value: dictionary of node/capacity for each child of parent
    all_paths = defaultdict(lambda: defaultdict(int))

    while bfs(graph, source, sink, flow_parents):
        path_flow = float("inf")
        s = sink
        path = {}
        while s != source:
            parent_node = flow_parents[s]
            for u, capacity in graph[parent_node]:
                if u == s:
                    path_flow = min(path_flow, capacity)
                    path[parent_node] = s
                    break
            s = flow_parents[s]

        for node, child in path.items():
            all_paths[node][child] += path_flow
            # print(f"adding path flow ({path_flow}) to all paths {node=}, {child=}")
        max_flow += path_flow
        v = sink
        while v != source:
            u = flow_parents[v]
            for i, (vertex, capacity) in enumerate(graph[u]):
                if vertex == v:
                    graph[u][i] = (vertex, capacity - path_flow)
                    break
            for i, (vertex, capacity) in enumerate(graph[v]):
                if vertex == u:
                    graph[v][i] = (vertex, capacity + path_flow)
                    break
            v = flow_parents[v]

    path_list = []
    for node in sorted(all_paths.keys()):
        for child in sorted(all_paths[node].keys()):
            path_list.append((node, child, all_paths[node][child]))
            if DEBUG:
                print(f"path from {node}->{child}, {capacity=}")

    if DEBUG:
        print(f"{max_flow=}\n")

    return max_flow, path_list

def bfs(graph, source, sink, flow_parents):
    """
    Use breadth first search to find a path from source to sink
    with any available capacity
    :param graph:
    :param source:
    :param sink:
    :param flow_parents:
    :return: True if there is some path with unused capacity
    """
    visited = set()
    queue = deque()
    queue.append(source)
    visited.add(source)

    while queue:
        u = queue.popleft()
        for v, capacity in graph[u]:
            if v not in visited and capacity > 0:
                queue.append(v)
                visited.add(v)
                flow_parents[v] = u
                if v == sink:
                    return True
    return False


# Explanation: The source node is 0, the sink node is 3;
# the capacity of edge (0->1) is 1; the capacity of edge (0->2) is 5; etc.
input_1 = (0, 3, [(0, 1, 1), (0, 2, 5), (1, 2, 1), (2, 3, 2), (1, 3, 6)])
# Explanation: "3" is the maximum flow;
# [(0, 1, 1), (0, 2, 2), (1, 3, 1), (2, 3, 2)]
# is an assignment of flows to edges.
expected_1 = (3, [(0, 1, 1), (0, 2, 2), (1, 3, 1), (2, 3, 2)])

input_2 = (0, 4,  [(0, 1, 2), (0, 3, 6), (1, 2, 3), (1, 3, 8), (1, 4, 5), (2, 4, 7), (3, 4, 9)])
expected_2 = (8, [(0, 1, 2), (0, 3, 6), (1, 4, 2), (3, 4, 6)])

# generated test cases, some hand editing
input_3 = (0, 7, [(6, 7, 8), (0, 2, 9), (3, 7, 7), (2, 7, 9), (3, 4, 6), (1, 4, 5),
                  (4, 7, 10), (1, 6, 10), (3, 5, 9), (1, 2, 8), (4, 5, 10), (0, 1, 3)])
expected_3 = (12, [(0, 1, 3), (0, 2, 9), (1, 4, 3), (2, 7, 9), (4, 7, 3)])

input_4 = (0, 7, [(6, 7, 8), (0, 2, 9), (3, 7, 7), (2, 7, 9), (3, 4, 6), (1, 4, 5), (4, 7, 10), (1, 6, 10), (3, 5, 9), (1, 2, 8), (4, 5, 10), (0, 4, 9), (0, 7, 10), (5, 6, 10), (1, 5, 10), (5, 7, 5)])
expected_4 = (28, [(0, 2, 9), (0, 4, 9), (0, 7, 10), (2, 7, 9), (4, 7, 9)])


class Testing(unittest.TestCase):

    def test_fat_pipes(self):
        result = Max_Flow_Fat(input_1)
        self.assertEqual(expected_1, result)

        result = Max_Flow_Fat(input_2)
        self.assertEqual(expected_2, result)

        result = Max_Flow_Fat(input_3)
        self.assertEqual(expected_3, result)

        result = Max_Flow_Fat(input_4)
        self.assertEqual(expected_4, result)

    def test_short_pipes(self):
        result = Max_Flow_Short(input_1)
        self.assertEqual(expected_1, result)

        result = Max_Flow_Short(input_2)
        self.assertEqual(expected_2, result)

        result = Max_Flow_Short(input_3)
        self.assertEqual(expected_3, result)

        result = Max_Flow_Short(input_4)
        self.assertEqual(expected_4, result)


if __name__ == '__main__':
    # generate testing sequences
    # from graph_gen import generate_seq
    # graph = generate_seq(8, 20, 0)
    # print(f"{graph=}")

    unittest.main()
