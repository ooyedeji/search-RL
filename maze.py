import pygame
from enum import Enum
from collections import namedtuple
import numpy as np
import random
from utils import flood_orthogonal


# Initialize the game engine
pygame.init()
font = pygame.font.Font("ubuntu.ttf", 20)

# Reward
REWARD_PATH = -0.05
REWARD_WALL = -0.75
REWARD_EXPLORED = -0.25
REWARD_GOAL = 1.00


class Point:
    """Represents a point in 2D space, typically on the game grid."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        return [self.x < other.x, self.y < other.y]

    def __gt__(self, other):
        return [self.x > other.x, self.y > other.y]

    def __sub__(self, other):
        return [self.x - other.x, self.y - other.y]

    def translate(self, dx, dy=0):
        """Translates the point by given differentials."""
        return Point(self.x + dx, self.y + dy)

    def distance_norm(self, other):
        """Calculates the Manhattan distance to another point."""
        return abs(self.x - other.x) + abs(self.y - other.y)

    def point_to_vector(self, scalar=1):
        return (self.x * scalar, self.y * scalar)


class Direction(Enum):
    """Represents the possible directions of movement."""

    RIGHT = (1, 0, 0, 0)
    LEFT = (0, 1, 0, 0)
    DOWN = (0, 0, 1, 0)
    UP = (0, 0, 0, 1)


class MazeStatus(Enum):
    LOSE = False
    WIN = True
    RUNNING = None


class Character(Enum):
    WALL = "#"
    SPACE = " "
    GOAL = "%"
    PLAYER = "O"

    def get_character_by_symbol(symbol: str):
        for member in Character.__members__.values():
            if member.value == symbol.upper():
                return member
        return Character.SPACE


Element = namedtuple("Element", ["character", "point"])


class Maze:
    GRID_SIZE = 40
    SPEED = 20

    def __init__(
        self,
        grid_path="grids/default.txt",
        random_initial=(False, False),
    ):
        # Initialize maze state
        self.grid_path = grid_path
        self.random_initial = np.atleast_1d(random_initial)
        self.reset()

        # Initialize screen
        screen_width = self.width * self.GRID_SIZE
        screen_height = self.height * self.GRID_SIZE
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption(self.__class__.__name__)
        self.clock = pygame.time.Clock()

        # Update UI
        self.update_ui()

    def reset(self):
        self.walls: list[Point] = []
        self.goal: Point | None = None
        self.player: Point | None = None
        self.explored: list[Point] = []

        # Read search problem from file
        with open(self.grid_path, mode="r") as file:
            for row_idx, line in enumerate(file.readlines()):
                for col_idx, symbol in enumerate(line.removesuffix("\n")):
                    character = Character.get_character_by_symbol(symbol)
                    point = Point(col_idx, row_idx)

                    if character == Character.WALL:
                        self.walls.append(point)
                    elif character == Character.PLAYER:
                        self.player = point
                    elif character == Character.GOAL:
                        self.goal = point

                self.width = max(col_idx + 1, getattr(self, "width", 0))
            self.height = row_idx + 1

        # Set maze info
        self.iteration = 0
        self.total_reward = 0

        if any(self.random_initial):
            path_coords = zip(*np.where(self.get_env_state() != REWARD_WALL))
            player_coord, goal_coord = random.sample(list(path_coords), 2)
        if self.random_initial[0]:
            self.player = Point(*player_coord)
        if self.random_initial[-1]:
            self.goal = Point(*goal_coord)

    def get_env_state(self, show_explored=True) -> np.ndarray:
        state = np.ones((self.width, self.height)) * REWARD_PATH

        for point in self.walls:
            state[*Point.point_to_vector(point)] = REWARD_WALL

        if show_explored:
            for point in self.explored:
                state[*Point.point_to_vector(point)] = REWARD_EXPLORED

        if self.player is not None:
            state[*Point.point_to_vector(self.player)] = 0.0

        if self.goal is not None:
            state[*Point.point_to_vector(self.goal)] = REWARD_GOAL

        return state

    def update_ui(self, points: list[Point] = []):
        # Colors palettes (RGB)
        BACKGROUND_COLOR = (40, 40, 40)
        GOAL_COLOR = (234, 25, 25)
        WALL_FILL_COLOR = (100, 124, 100)
        WALL_EDGE_COLOR = (57, 74, 57)
        PLAYER_COLOR = (245, 223, 4)
        EXPLORED_COLOR = (50, 50, 50)
        EXTRA_COLOR = (247, 234, 101)

        # Reset display
        self.screen.fill(BACKGROUND_COLOR)

        # Draw wall
        for idx, point in enumerate(self.walls):
            x, y = Point.point_to_vector(point, self.GRID_SIZE)
            rect = pygame.Rect(x, y, self.GRID_SIZE, self.GRID_SIZE)
            width = self.GRID_SIZE // 10
            pygame.draw.rect(self.screen, WALL_FILL_COLOR, rect)
            pygame.draw.rect(self.screen, WALL_EDGE_COLOR, rect, width)

        # Draw goal
        x, y = Point.point_to_vector(self.goal, self.GRID_SIZE)
        center = (x + self.GRID_SIZE / 2, y + self.GRID_SIZE / 2)
        radius = self.GRID_SIZE // 3
        pygame.draw.circle(self.screen, GOAL_COLOR, center, radius)

        # Draw explored
        for idx, point in enumerate(self.explored):
            if point in self.explored[:idx]:
                continue
            x, y = Point.point_to_vector(point, self.GRID_SIZE)
            border = self.GRID_SIZE / 10
            size = self.GRID_SIZE - 2 * border
            rect = pygame.Rect(x + border, y + border, size, size)
            pygame.draw.rect(self.screen, EXPLORED_COLOR, rect)

        # Draw player
        x, y = Point.point_to_vector(self.player, self.GRID_SIZE)
        center = (x + self.GRID_SIZE / 2, y + self.GRID_SIZE / 2)
        radius = self.GRID_SIZE // 4
        pygame.draw.circle(self.screen, PLAYER_COLOR, center, radius)

        # Draw given points
        for idx, point in enumerate(points):
            x, y = Point.point_to_vector(point, self.GRID_SIZE)
            border = self.GRID_SIZE / 3
            size = self.GRID_SIZE - 2 * border
            rect = pygame.Rect(x + border, y + border, size, size)
            pygame.draw.rect(self.screen, EXTRA_COLOR, rect)

        # Update the screen
        pygame.display.flip()

    def _get_direction_from_key(self, event: pygame.event.Event):
        if event.key == pygame.K_LEFT:
            new_direction = Direction.LEFT
        elif event.key == pygame.K_RIGHT:
            new_direction = Direction.RIGHT
        elif event.key == pygame.K_UP:
            new_direction = Direction.UP
        elif event.key == pygame.K_DOWN:
            new_direction = Direction.DOWN
        else:
            new_direction = None

        return new_direction

    def get_movement_vector(self, direction: Direction = None):
        dx, dy = 0, 0
        if direction == Direction.UP:
            dy = -1
        elif direction == Direction.DOWN:
            dy = 1
        elif direction == Direction.LEFT:
            dx = -1
        elif direction == Direction.RIGHT:
            dx = 1

        return dx, dy

    def _move_player(self, direction: Direction = None):
        # Calculate player movement, reward, and track prior position
        dx, dy = self.get_movement_vector(direction)
        prior_point = self.player
        next_point = Point.translate(self.player, dx=dx, dy=dy)
        try:
            reward = self.get_env_state()[*Point.point_to_vector(next_point)]
        except IndexError:
            reward = REWARD_WALL

        # Handle collision & movement outside grid, track explored positions
        if not (self.is_wall(next_point) or self.is_ghost(next_point)):
            self.player = next_point
            self.explored.append(prior_point)

        return reward

    def is_target(self, point: Point = None):
        point = self.player if point is None else point
        return point == self.goal

    def is_explored(self, point: Point = None):
        point = self.player if point is None else point
        return point in list(self.explored)

    def is_wall(self, point: Point = None):
        point = self.player if point is None else point
        return point in self.walls

    def is_ghost(self, point: Point = None):
        point = self.player if point is None else point
        out_x = point.x < 0 or point.x >= self.width
        out_y = point.y < 0 or point.y >= self.height
        return out_x or out_y

    def is_access(self, direction: Direction, point: Point = None):
        # Default point is the player if not provided
        point = self.player if point is None else point

        # Retrieve the current environment state
        env_state = self.get_env_state(show_explored=False)

        # Mark goal and player points in the environment state
        goal_vector = Point.point_to_vector(self.goal)
        point_vector = Point.point_to_vector(point)
        env_state[*goal_vector] = REWARD_PATH
        env_state[*point_vector] = REWARD_WALL

        # Calculate the coordinates for the next point
        dx, dy = self.get_movement_vector(direction)
        next_point = Point.translate(self.player, dx=dx, dy=dy)

        # If the next point is a "ghost", the value is 0 (not accessible)
        if self.is_ghost(next_point):
            return 0

        # Perform flood-fill from the goal to determine the accessibility
        flood_filled_state = flood_orthogonal(env_state, goal_vector)

        # Return the value at the next point (either 0 or a valid path value)
        return int(flood_filled_state[*Point.point_to_vector(next_point)])

    def _get_next_point(self, direction: Direction, point: Point = None):
        """Helper method to calculate the next point based on direction."""
        point = self.player if point is None else point
        dx, dy = self.get_movement_vector(direction)
        return Point.translate(point, dx, dy)

    def get_delta_heuristic(self, direction: Direction, point: Point = None):
        point = self.player if point is None else point
        next_point = self._get_next_point(direction, point)
        h1 = point.distance_norm(self.goal)
        h2 = next_point.distance_norm(self.goal)
        return h2 - h1

    def will_hit_wall(self, direction: Direction, point: Point = None):
        next_point = self._get_next_point(direction, point)
        return self.is_wall(next_point)

    def will_reach_target(self, direction: Direction, point: Point = None):
        next_point = self._get_next_point(direction, point)
        return self.is_target(next_point)

    def will_leave_area(self, direction: Direction, point: Point = None):
        next_point = self._get_next_point(direction, point)
        return self.is_ghost(next_point)

    def will_visit_explored(self, direction: Direction, point: Point = None):
        next_point = self._get_next_point(direction, point)
        return self.is_explored(next_point)

    def _get_status(self):
        if self.is_target():
            status = MazeStatus.WIN
        elif self.iteration > (self.get_env_state() == REWARD_PATH).sum():
            status = MazeStatus.LOSE
        else:
            status = MazeStatus.RUNNING

        return status

    def play_step(self, direction: Direction = None):
        # Get user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            elif event.type == pygame.KEYDOWN and direction is None:
                direction = self._get_direction_from_key(event)

        # If no direction is provided
        if direction is None:
            return 0, MazeStatus.RUNNING

        # Increment the iteration count
        self.iteration += 1

        # Move: Update player position, get reward.
        reward = self._move_player(direction)
        self.total_reward += reward

        # Determine the current status of the environment
        status = self._get_status()

        # Update UI
        self.update_ui()
        self.clock.tick(self.SPEED)

        return reward, status


if __name__ == "__main__":
    maze = Maze(grid_path="grids/default.txt", random_initial=False)
    while True:
        reward, status = maze.play_step()

        if status.value is not None:
            break

    pygame.quit()
