"""Grid-based maze environment for Q-learning experiments."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple


Position = Tuple[int, int]


class MazeEnv:
    """Deterministic maze with terminal goal and obstacle states."""

    ACTIONS: Dict[int, Position] = {
        0: (-1, 0),  # up
        1: (1, 0),   # down
        2: (0, -1),  # left
        3: (0, 1),   # right
    }
    ACTION_NAMES = ["Up", "Down", "Left", "Right"]

    def __init__(self, size: int = 6, obstacles: Optional[Iterable[Position]] = None) -> None:
        if size < 2:
            raise ValueError("Maze size must be at least 2.")

        self.size = size
        self.start_state: Position = (0, 0)
        self.goal_state: Position = (size - 1, size - 1)
        self.obstacles = set(obstacles) if obstacles is not None else set(self._default_obstacles())

        if self.start_state in self.obstacles or self.goal_state in self.obstacles:
            raise ValueError("Start and goal positions cannot be obstacles.")

        self.n_states = size * size
        self.n_actions = len(self.ACTIONS)
        self.agent_position: Position = self.start_state

    def _default_obstacles(self) -> List[Position]:
        """Obstacle layout that still leaves a valid path to the goal."""
        candidates = [(1, 1), (1, 3), (2, 3), (3, 1), (3, 4), (4, 2)]
        return [
            cell
            for cell in candidates
            if cell[0] < self.size and cell[1] < self.size and cell not in {self.start_state, self.goal_state}
        ]

    def reset(self) -> int:
        """Reset the agent to the starting state."""
        self.agent_position = self.start_state
        return self.state_to_index(self.agent_position)

    def state_to_index(self, state: Position) -> int:
        row, col = state
        return row * self.size + col

    def index_to_state(self, index: int) -> Position:
        return divmod(index, self.size)

    def is_valid_position(self, position: Position) -> bool:
        row, col = position
        return 0 <= row < self.size and 0 <= col < self.size

    def step(self, action: int) -> Tuple[int, int, bool, Dict[str, Position]]:
        """Apply an action and return next_state, reward, done, info."""
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action {action}. Valid actions are 0 to {self.n_actions - 1}.")

        row_delta, col_delta = self.ACTIONS[action]
        next_position = (
            self.agent_position[0] + row_delta,
            self.agent_position[1] + col_delta,
        )

        if not self.is_valid_position(next_position):
            next_position = self.agent_position

        self.agent_position = next_position

        if next_position in self.obstacles:
            return self.state_to_index(next_position), -100, True, {"position": next_position}
        if next_position == self.goal_state:
            return self.state_to_index(next_position), 100, True, {"position": next_position}
        return self.state_to_index(next_position), -1, False, {"position": next_position}

    def render_path(self, path: List[Position]) -> str:
        """Return a text visualization of a path through the maze."""
        grid = [["." for _ in range(self.size)] for _ in range(self.size)]

        for row, col in self.obstacles:
            grid[row][col] = "X"

        for row, col in path:
            if (row, col) not in {self.start_state, self.goal_state} and (row, col) not in self.obstacles:
                grid[row][col] = "*"

        start_row, start_col = self.start_state
        goal_row, goal_col = self.goal_state
        grid[start_row][start_col] = "S"
        grid[goal_row][goal_col] = "G"

        return "\n".join(" ".join(row) for row in grid)
