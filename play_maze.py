import torch
import pygame
from maze import Maze, MazeStatus
from agent import Agent

# Initialize maze board and AI agent
maze = Maze(grid_path="grids/default.txt", random_initial=False)
agent = Agent()

# Initialize the model
model = agent.model

# Load the saved model weights
model.load_state_dict(torch.load("models/model_0.pth"))
model.eval()

while True:
    # Get current state
    state = agent.get_state(maze)

    # Get action from model
    action_id = agent.get_action(state, explore=False)

    # Advance game with action
    reward, status = maze.play_step(agent.directions[action_id])

    if status != MazeStatus.RUNNING:
        break

print(f"Score: {maze.total_reward:.2f}")
pygame.quit()
