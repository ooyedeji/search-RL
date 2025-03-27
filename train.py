import numpy as np
from maze import Maze, MazeStatus
from agent import Agent


def qtrain():
    # Track board metrics
    scores = []
    mean_scores = []
    win_streak = 0

    # Initialize maze board and AI agent
    maze = Maze(random_initial=True)
    agent = Agent()

    while True:
        # Get current state
        state = agent.get_state(maze)

        # Get action index from model
        action_id = agent.get_action(state)

        # Advance board with action
        reward, status = maze.play_step(agent.directions[action_id])
        next_state = agent.get_state(maze)

        # Memorize outcomes
        agent.memorize(state, action_id, status, reward, next_state)

        # Train on short memory
        agent.train_short_memory()

        if status.value is not None:
            # Train on long memory and update maze episode count
            agent.train_long_memory()
            agent.n_episode += 1

            # Update scores
            scores.append(maze.total_reward)
            mean_score = np.mean(scores[-50:])
            mean_scores.append(mean_score)
            if status == MazeStatus.WIN:
                win_streak += 1
            else:
                win_streak = 0

            # Reset board
            maze.reset()

            # Save best model
            if win_streak > 0 and win_streak % 50 == 0:
                agent.model.save(f"models/model_0.pth")

            # Print progress
            print(
                f"Episode: {agent.n_episode:>5}",
                f"-- Score: {scores[-1]:.1f}",
                f"-- mean_score: {mean_scores[-1]:.1f}",
                f"-- Record: {max(scores):.1f}",
                f"-- win_streak: {win_streak}",
            )

        # Stop if the mean score continuously decreases
        if win_streak >= 200:
            print("QNet training completed.")
            agent.model.save(f"models/model_1.pth")
            break


if __name__ == "__main__":
    qtrain()
