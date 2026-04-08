"""Q-learning agent implementation for the maze environment."""

from __future__ import annotations

import numpy as np


class QLearningAgent:
    """Tabular Q-learning agent with epsilon-greedy exploration."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        min_epsilon: float = 0.01,
        decay_rate: float = 0.995,
        random_seed: int = 42,
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.rng = np.random.default_rng(random_seed)

        self.q_table = np.zeros((state_size, action_size), dtype=float)

    def choose_action(self, state: int) -> int:
        """Choose an action using an epsilon-greedy policy."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_size))
        return self.greedy_action(state)

    def greedy_action(self, state: int) -> int:
        """Choose the best current action for a state."""
        state_values = self.q_table[state]
        best_actions = np.flatnonzero(state_values == np.max(state_values))
        return int(self.rng.choice(best_actions))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        """Apply the Q-learning update rule."""
        next_max = 0.0 if done else np.max(self.q_table[next_state])
        target = reward + self.gamma * next_max
        current_value = self.q_table[state, action]
        self.q_table[state, action] = current_value + self.alpha * (target - current_value)

    def decay_epsilon(self) -> None:
        """Reduce exploration after an episode."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
