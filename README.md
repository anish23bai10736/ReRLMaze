# Q-Learning Maze Solver

This project implements a modular Q-learning based maze solver in Python for a reinforcement learning academic submission. The agent learns to move in a grid world from the start cell `(0, 0)` to the goal cell `(size-1, size-1)` while avoiding obstacle cells.

## Project Structure

```text
RL_Maze_Project/
|-- maze_env.py
|-- q_learning_agent.py
|-- train.py
|-- experiments.py
|-- plots.py
`-- README.md
```

## Environment Details

- Grid size: configurable, default `6 x 6`
- Start state: `(0, 0)`
- Goal state: `(5, 5)` for the default maze
- Actions: up, down, left, right
- Movement: deterministic
- Rewards:
  - Goal: `+100`
  - Obstacle: `-100` and the episode ends
  - Normal step: `-1`

## Q-Learning Overview

Q-learning is a model-free reinforcement learning algorithm that learns the value of taking an action in a particular state. The update rule used in this project is:

```text
Q(s, a) = Q(s, a) + alpha * [reward + gamma * max(Q(s', a')) - Q(s, a)]
```

Where:

- `alpha` is the learning rate
- `gamma` is the discount factor
- `epsilon` controls exploration in the epsilon-greedy policy

The agent explores randomly with probability `epsilon`, otherwise it chooses the action with the highest Q-value.

## Experiments Implemented

Two training settings are compared:

1. Fixed epsilon: `epsilon = 0.1`
2. Decaying epsilon: starts at `1.0` and decays after each episode using `epsilon *= 0.995`

For both settings, the project tracks:

- Total reward per episode
- Steps per episode
- Epsilon values per episode
- Convergence behavior

The convergence comparison uses a simple criterion: the first episode after which the agent reaches the goal successfully for `20` consecutive episodes.

## How to Run

Make sure Python has the required dependencies installed:

```bash
pip install numpy matplotlib
```

Run the full experiment comparison:

```bash
python experiments.py
```

Optional: run a single training demonstration:

```bash
python train.py
```

## Output Generated

After running `experiments.py`, the project:

- Prints training progress every 50 episodes
- Prints the final Q-table for each experiment
- Prints the learned path from the start state to the goal
- Shows a text-based grid visualization of the learned path
- Saves the following plots in the project directory:
  - `reward_vs_episode.png`
  - `steps_vs_episode.png`
  - `epsilon_decay.png`

## Expected Observations

- The fixed epsilon setting usually learns steadily but keeps exploring throughout training.
- The decaying epsilon setting explores more in the early episodes and gradually exploits more later.
- In many runs, decaying epsilon converges faster because it balances initial exploration with later exploitation.
- Rewards generally improve over time, while the number of steps required to reach the goal tends to decrease.

## Submission Note

The project was created inside the required directory and kept clean with only the requested source files. For final upload, use a top-level folder name in the format `510_<Regnumbers>`. In this local build, the folder name `510_REGNUMBERS` is used because angle brackets are not valid characters in Windows folder names.
