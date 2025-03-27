import torch
import random
import numpy as np
from collections import deque
from maze import Maze, Direction
from model import Linear_QNet, QTrainer


class Agent:
    GAMMA = 0.95
    LR = 0.001
    MAX_MEMORY = 100_000
    BATCH_SIZE = 1000

    def __init__(self):
        self.n_episode = 1
        self.memory = deque(maxlen=self.MAX_MEMORY)
        self.model = Linear_QNet(28, (64, 16), 4)
        self.trainer = QTrainer(self.model, lr=self.LR, gamma=self.GAMMA)

        # Possible movement directions
        self.directions = list(Direction.__members__.values())

    @property
    def epsilon(self) -> float:
        x = self.n_episode
        epsilon_end = 0.01
        theta1, theta2, theta3 = 1 - epsilon_end, 0.05, 100
        beta = 1
        y = theta1 / (1 + beta * np.exp(-theta2 * (x - theta3))) ** (1 / beta)

        return 1 - y

    def get_state(self, maze: Maze):
        state = [
            [maze.is_access(d) for d in self.directions],
            [maze.will_hit_wall(d) for d in self.directions],
            [maze.will_leave_area(d) for d in self.directions],
            [maze.will_reach_target(d) for d in self.directions],
            [maze.will_visit_explored(d) for d in self.directions],
            [maze.get_delta_heuristic(d) for d in self.directions],
            (maze.player > maze.goal) + (maze.player < maze.goal),
        ]
        return np.array(sum(state, []), dtype=float)

    def memorize(self, state, action, status, reward, next_state):
        self.memory.append((state, action, status, reward, next_state))

    def train_long_memory(self):
        batch_size = min(self.BATCH_SIZE, len(self.memory))
        sample = random.sample(self.memory, batch_size)
        self.trainer.train_step(*zip(*sample))

    def train_short_memory(self):
        self.trainer.train_step(*self.memory[-1])

    def get_action(self, state, explore=True):
        if self.epsilon > random.random() and explore:
            # Exploration
            action_id = random.randint(0, len(self.directions) - 1)
        else:
            # Exploitation
            state = torch.tensor(np.array(state), dtype=torch.float)
            state = state.unsqueeze(0)
            action_id = torch.argmax(self.model(state)).item()

        return action_id
