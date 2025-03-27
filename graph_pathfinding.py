from abc import ABC, abstractmethod
from maze import Maze, Direction, Point
import random
from enum import Enum


class Node:

    def __init__(self, point, parent=None, g_cost=None, h_cost=None):
        self.point = point
        self.parent = parent
        self.g_cost = g_cost or 0
        self.h_cost = h_cost or 0

    @property
    def f_cost(self):
        return self.g_cost + self.h_cost

    def __eq__(self, other):
        return self.point == other.point


class DataStructure(ABC):

    def __init__(self, data: list[Node] = []):
        self.data = data if isinstance(data, list) else [data]

    def __len__(self):
        return len(self.data)

    def is_empty(self):
        return len(self.data) == 0

    def enqueue(self, item):
        self.data.append(item)

    @abstractmethod
    def dequeue(self):
        pass


class Stack(DataStructure):

    def __init__(self, data: list[Node] = []):
        super().__init__(data)

    def dequeue(self):
        item = self.data.pop(-1)
        return item


class Queue(DataStructure):

    def __init__(self, data: list[Node] = []):
        super().__init__(data)

    def dequeue(self):
        item = self.data.pop(0)
        return item


class PriorityQueue(DataStructure):

    def __init__(self, data: list[Node] = []):
        super().__init__(data)

    def dequeue(self):
        f_costs = [node.f_cost for node in self.data]
        item = self.data.pop(f_costs.index(min(f_costs)))
        return item


class Algorithm(Enum):
    AStar = 0
    BFS = 1
    DFS = 2

    def get_algorithm_by_value(value: int | str):
        for member in Algorithm.__members__.values():
            if isinstance(value, int) and member.value == value:
                return member
            elif isinstance(value, str):
                if member.name.lower() == value.lower():
                    return member
        return Algorithm.AStar


class Search:
    def __init__(self, maze: Maze, algorithm: Algorithm = 0):
        self.maze = maze
        self.explored_nodes: list[Node] = []
        self.directions = list(Direction.__members__.values())
        self.algorithm = Algorithm.get_algorithm_by_value(algorithm)

        self.start = Node(self.maze.player)

        if self.algorithm == Algorithm.BFS:
            self.frontier = Queue(self.start)
        elif self.algorithm == Algorithm.DFS:
            self.frontier = Stack(self.start)
        elif self.algorithm == Algorithm.AStar:
            self.frontier = PriorityQueue(self.start)

    def expand(self, node: Node):
        random.shuffle(self.directions)
        for direction in self.directions:
            dx, dy = self.maze.get_movement_vector(direction=direction)
            next_point = Point.translate(node.point, dx, dy)
            is_path = not self.maze.is_wall(next_point)
            is_exist = not self.maze.is_ghost(next_point)
            is_unexplored = not self.maze.is_explored(next_point)

            if all((is_path, is_exist, is_unexplored)):
                if self.algorithm in [Algorithm.BFS, Algorithm.DFS]:
                    h_cost = None
                    g_cost = None

                elif self.algorithm in [Algorithm.AStar]:
                    h_cost = Point.distance_norm(next_point, self.maze.goal)
                    g_cost = node.g_cost + 1

                next_node = Node(next_point, node.point, g_cost, h_cost)
                self.frontier.enqueue(next_node)

    def run(self):
        while True:
            # Expand search frontier
            if not self.frontier.is_empty():
                node = self.frontier.dequeue()
                if node not in self.explored_nodes:
                    self.expand(node)
                else:
                    continue
            else:
                print("Search unsuccessful!")
                break

            # Update explored points
            self.explored_nodes.append(node)

            # Update maze state and UI
            self.maze.player = node.point
            self.trace_solution()
            self.maze.update_ui(points=self.track)
            self.maze.clock.tick(5)
            self.maze.explored = [node.point for node in self.explored_nodes]

            if node.point == self.maze.goal:
                self.trace_solution()
                print("Goal reached!")
                break

    def trace_solution(self):
        self.track = []
        parent = self.explored_nodes[-1].parent
        for i, node in enumerate(reversed(self.explored_nodes)):
            if i > 0 and node.point == parent:
                self.track.append(node.point)
                parent = node.parent


if __name__ == "__main__":
    maze = Maze(grid_path="grids/default.txt")
    search = Search(maze=maze, algorithm=0)

    search.run()
