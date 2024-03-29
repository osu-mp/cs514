"""
Matthew Pacey
CS 514 - HW4 - Graphs
"""
from queue import SimpleQueue
import unittest

# set to True to print board in one line (probably easier for grading)
# False to print a 3x3 of the board (easier debugging)
SIMPLE_PRINT = True

def ShortestPath(goal, init_states):
    """
    Use BFS to solve the 8-puzzle (3x3 grid with empty slot)

    :param goal: Array of desired board config, e.g. [1, 2, 3, 4, 5, 6, 7, 8, 0]
    :param init_states: Array of initial states (target the goal state)
        There is only 1 goal but can be multiple initial states
    :return: List of minimum number of moves to move from each init state to the goal
    """
    # minimum number of moves to solve each input init state (depth in BFS tree)
    shortest_path_lens = []

    if not valid_config(goal):
        for i in range(len(init_states)):
            shortest_path_lens.append(None)
        return shortest_path_lens

    print(f"{visualize(goal, 'Goal')}")

    # iterate over each initial state, finding a path to the single goal
    for init_state in init_states:
        if not valid_config(init_state):
            shortest_path_lens.append(None)
            continue
        init_key = stringify(init_state)

        visualize(goal, "Goal")
        visualize(init_state, "Init")

        # key = str representation of board, value = min depth in BFS tree
        visited = {}

        # create a queue of nodes to visit, start with goal (root)
        queue = SimpleQueue()
        queue.put(goal)
        solved = False

        # iterate over all nodes in the graph (added in BFS manner to queue)
        while not queue.empty():
            # pop off the first node
            vertex = queue.get()
            vertex_str = stringify(vertex)
            if vertex_str not in visited:
                parent_depth = 0
            else:
                parent_depth = visited[vertex_str]

            # if this node matches our init state, its depth represents the minimum
            # number of moves to transition from goal to start
            if vertex_str == init_key:
                shortest_path_lens.append(parent_depth)
                label = f"\tInit State (min. moves={parent_depth})"
                print(f"{visualize(init_state, label)}")
                solved = True
                break

            # else add all valid, unvisited children of this node to the queue
            child_depth = parent_depth + 1
            visualize(vertex, "Vertex")
            children = get_permutations(vertex)

            for child in children:
                visualize(child, "Child")
                key = stringify(child)

                # add any unvisited children to the queue
                if key not in visited:
                    visited[key] = child_depth
                    queue.put(child)

        if not solved:
            shortest_path_lens.append(None)
            print(f"{visualize(init_state, '    Init state (unsolvable)')}")
    return shortest_path_lens

def stringify(nums):
    # convert a list of numbers into a string to use as a hash key
    return ' '.join(map(str, nums))

def valid_config(nums):
    if len(nums) != 9:
        print("Invalid input, should be numbers 0 through 8")
        return False
    for i in range(9):
        if i not in nums:
            print(f"Invalid input, missing number {i}")
            return False


    return True

def create_copy_with_swap(nums, start, end):
    """
    Create a new grid as a copy of the parent vertex
    Swap two of the values (the 0 will be swapped in all directions it can legally go)
    :param nums:
    :param start:
    :param end:
    :return:
    """
    new_nums = nums.copy()
    new_nums[start], new_nums[end] = new_nums[end], new_nums[start]
    return new_nums


def get_permutations_simple(nums):
    pos = nums.index(0)
    perms = []
    for index in [
        pos - 1, pos + 1,   # left, right
        pos - 3, pos + 3    # up, down
    ]:
        if index >= 0 and index <= 8:
            perms.append(create_copy_with_swap(nums, pos, index))

    return perms

def get_permutations(nums):
    """
    Brute force way to find all child permutations for a given parent board config
    Find the 0 (empty slot) and move it up, down, left, right (whichever are valid)
    :param nums:
    :return:
    """
    pos = nums.index(0)
    perms = []
    # for index in [
    #     pos - 1, pos + 1,   # left, right
    #     pos - 3, pos + 3    # up, down
    # ]:
    #     if index >= 0 and index <= 8:
    #         perms.append(create_copy_with_swap(nums, pos, index))
    if pos == 0:  # top left
        perms.append(create_copy_with_swap(nums, pos, 1))  # right
        perms.append(create_copy_with_swap(nums, pos, 3))  # down
    elif pos == 1:  # top mid
        perms.append(create_copy_with_swap(nums, pos, 0))  # left
        perms.append(create_copy_with_swap(nums, pos, 2))  # right
        perms.append(create_copy_with_swap(nums, pos, 4))  # down
    elif pos == 2:  # top right
        perms.append(create_copy_with_swap(nums, pos, 1))  # left
        perms.append(create_copy_with_swap(nums, pos, 5))  # down
    elif pos == 3:  # mid left
        perms.append(create_copy_with_swap(nums, pos, 0))  # up
        perms.append(create_copy_with_swap(nums, pos, 4))  # right
        perms.append(create_copy_with_swap(nums, pos, 6))  # down
    elif pos == 4:  # center
        perms.append(create_copy_with_swap(nums, pos, 1))  # up
        perms.append(create_copy_with_swap(nums, pos, 5))  # right
        perms.append(create_copy_with_swap(nums, pos, 7))  # down
        perms.append(create_copy_with_swap(nums, pos, 3))  # left
    elif pos == 5:  # mid right
        perms.append(create_copy_with_swap(nums, pos, 2))  # up
        perms.append(create_copy_with_swap(nums, pos, 4))  # left
        perms.append(create_copy_with_swap(nums, pos, 8))  # down
    elif pos == 6:  # bottom left
        perms.append(create_copy_with_swap(nums, pos, 3))  # up
        perms.append(create_copy_with_swap(nums, pos, 7))  # right
    elif pos == 7:  # bottom mid
        perms.append(create_copy_with_swap(nums, pos, 6))  # left
        perms.append(create_copy_with_swap(nums, pos, 4))  # up
        perms.append(create_copy_with_swap(nums, pos, 8))  # right
    elif pos == 8:  # bottom right
        perms.append(create_copy_with_swap(nums, pos, 7))  # left
        perms.append(create_copy_with_swap(nums, pos, 5))  # up
    else:
        print("invalid index")

    return perms


def visualize(nums, title="Current:"):
    if SIMPLE_PRINT:
        str = f"{title} {nums}"
        # print(str)
        return str

    grid_size = 3

    # Debug routine to show the current board config
    str = f"{title}\n"
    for i in range(len(nums)):
        str += f"{nums[i]} "
        if (i + 1) % grid_size == 0:
            str += "\n"
    # print(str)
    return str


def permutation_visualizer():
    nums = [0, 2, 3, 4, 5, 6, 7, 8, 9, 1]
    visualize(nums, "top_left")
    for perm in get_permutations(nums):
        visualize(perm, "top_left_perm")

    nums = [5, 2, 3, 4, 0, 6, 7, 8, 9, 1]
    visualize(nums, "center")
    for perm in get_permutations(nums):
        visualize(perm, "center_perm")


class Testing(unittest.TestCase):
    def test_solved(self):
        goal = [1, 2, 3, 8, 0, 4, 7, 6, 5]
        init_states = [goal]
        exp = [0]
        self.assertEqual(exp, ShortestPath(goal, init_states))

        init_states = [goal, goal, goal]
        exp = [0, 0, 0]
        self.assertEqual(exp, ShortestPath(goal, init_states))

    def test_single_move(self):
        goal = [1, 2, 3, 8, 0, 4, 7, 6, 5]
        init_states = [[1, 0, 3, 8, 2, 4, 7, 6, 5]]
        exp = [1]
        self.assertEqual(exp, ShortestPath(goal, init_states))

    def test_small_moves(self):
        goal = [1, 2, 3, 8, 0, 4, 7, 6, 5]
        init_states = [[1, 2, 3, 8, 6, 4, 7, 5, 0]]
        exp = [2]
        self.assertEqual(exp, ShortestPath(goal, init_states))


        goal = [1, 2, 3, 0, 4, 6, 7, 5, 8]
        init_states = [
            [1, 2, 3, 4, 5, 6, 7, 8, 0],
            [1, 2, 3, 4, 5, 0, 7, 8, 6],
            [1, 2, 3, 4, 0, 5, 7, 8, 6],
        ]
        exp = [3, 4, 5]
        self.assertEqual(exp, ShortestPath(goal, init_states))

        goal = [1, 2, 3, 8, 0, 4, 7, 6, 5]
        init_states = [
            [3, 6, 4, 0, 1, 2, 8, 7, 5],
            [6, 0, 4, 3, 1, 2, 8, 7, 5],
            [6, 6, 0, 3, 1, 2, 8, 7, 5], # unsolvable, expect None
        ]
        exp = [11, 13, None]
        self.assertEqual(exp, ShortestPath(goal, init_states))

    def test_invalid_inputs(self):
        goal = [1, ]
        init_states = [
            [3, 6, 4, 0, 1, 2, 8, 7, 5],
            [3, 6, 4, 0, 1, 2, 8, 7, 5],
        ]
        exp = [None, None]
        self.assertEqual(exp, ShortestPath(goal, init_states))

        [3, 6, 4, 0, 1, 2, 8, 7, 5],
        init_states = [
            [3, 6, 4, 0, 1, 2, 8, 7, 0],
            [3, 6, 4, 0, 1, 2, 8, 7, 9],
        ]
        exp = [None, None]
        self.assertEqual(exp, ShortestPath(goal, init_states))

    def test_demo(self):
        goal = [1, 2, 3, 8, 0, 4, 7, 6, 5]
        init_states = [
            [1, 2, 3, 4, 5, 6, 8, 7, 0],
            [2, 8, 1, 4, 6, 3, 0, 7, 5]
        ]
        exp = [8, 12]
        self.assertEqual(exp, ShortestPath(goal, init_states))

        goal = [4, 1, 2, 0, 8, 7, 6, 3, 5]
        init_states = [
            [1, 2, 3, 4, 5, 6, 7, 8, 0]
        ]
        exp = [17]
        self.assertEqual(exp, ShortestPath(goal, init_states))


if __name__ == '__main__':
    unittest.main()