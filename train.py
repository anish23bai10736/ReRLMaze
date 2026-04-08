"""Reusable training helpers for the maze Q-learning project."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from maze_env import MazeEnv
from q_learning_agent import QLearningAgent


def train_agent(
    env: MazeEnv,
    agent: QLearningAgent,
    episodes: int = 700,
    max_steps: Optional[int] = None,
    decay_epsilon: bool = False,
    progress_interval: int = 50,
) -> Dict[str, List[float]]:
    """Train an agent and return episode-wise training history."""
    max_steps = max_steps or (env.size * env.size * 4)

    history: Dict[str, List[float]] = {
        "rewards": [],
        "steps": [],
        "epsilons": [],
        "successes": [],
    }

    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        episode_steps = 0
        success = 0

        for step in range(1, max_steps + 1):
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)

            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            episode_steps = step

            if done:
                success = int(info["position"] == env.goal_state)
                break

        history["rewards"].append(total_reward)
        history["steps"].append(episode_steps)
        history["epsilons"].append(agent.epsilon)
        history["successes"].append(success)

        if decay_epsilon:
            agent.decay_epsilon()

        if episode == 1 or episode % progress_interval == 0 or episode == episodes:
            print(
                f"Episode {episode:>3}/{episodes} | "
                f"Reward: {total_reward:>6.1f} | "
                f"Steps: {episode_steps:>2} | "
                f"Epsilon: {history['epsilons'][-1]:.4f}"
            )

    return history


def extract_optimal_path(
    env: MazeEnv,
    agent: QLearningAgent,
    max_steps: Optional[int] = None,
) -> Tuple[List[Tuple[int, int]], bool]:
    """Follow the learned greedy policy from start to goal."""
    max_steps = max_steps or (env.size * env.size * 4)
    state = env.reset()
    path = [env.start_state]
    visited = {env.start_state}

    for _ in range(max_steps):
        action = agent.greedy_action(state)
        next_state, _, done, info = env.step(action)
        position = info["position"]
        path.append(position)

        if done:
            return path, position == env.goal_state

        if position in visited:
            return path, False

        visited.add(position)
        state = next_state

    return path, False


def moving_average(values: List[float], window: int = 20) -> np.ndarray:
    """Simple moving average used to smooth experiment curves."""
    if not values:
        return np.array([])

    values_array = np.asarray(values, dtype=float)
    if len(values_array) < window:
        return values_array

    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values_array, kernel, mode="valid")


def convergence_episode(successes: List[int], window: int = 20) -> Optional[int]:
    """Return the first episode index where the agent is consistently successful."""
    if len(successes) < window:
        return None

    success_array = np.asarray(successes, dtype=int)
    rolling_success = np.convolve(success_array, np.ones(window, dtype=int), mode="valid")
    converged_indices = np.where(rolling_success == window)[0]

    if converged_indices.size == 0:
        return None
    return int(converged_indices[0] + window)


def print_q_table(agent: QLearningAgent) -> None:
    """Pretty-print the learned Q-table."""
    np.set_printoptions(precision=2, suppress=True)
    print(agent.q_table)


if __name__ == "__main__":
    environment = MazeEnv()
    learning_agent = QLearningAgent(
        state_size=environment.n_states,
        action_size=environment.n_actions,
        epsilon=0.1,
    )

    training_history = train_agent(environment, learning_agent, episodes=700, decay_epsilon=False)
    learned_path, reached_goal = extract_optimal_path(environment, learning_agent)

    print("\nSingle run complete.")
    print(f"Reached goal with greedy policy: {reached_goal}")
    print(f"Learned path: {learned_path}")
    print("\nPath visualization:")
    print(environment.render_path(learned_path))
    print("\nFinal Q-table:")
    print_q_table(learning_agent)
