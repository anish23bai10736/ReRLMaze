"""Run fixed and decaying epsilon Q-learning experiments on the maze."""

from __future__ import annotations

import os
from typing import Dict

import numpy as np

from maze_env import MazeEnv
from plots import output_path, plot_comparison_metric, plot_epsilon_decay
from q_learning_agent import QLearningAgent
from train import convergence_episode, extract_optimal_path, print_q_table, train_agent


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
EPISODES = 700


def run_experiment(
    label: str,
    epsilon: float,
    decay_epsilon: bool,
    alpha: float = 0.1,
    gamma: float = 0.95,
    decay_rate: float = 0.995,
) -> Dict[str, object]:
    """Train one experiment setting and return its outputs."""
    env = MazeEnv(size=6)
    agent = QLearningAgent(
        state_size=env.n_states,
        action_size=env.n_actions,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        min_epsilon=0.01,
        decay_rate=decay_rate,
        random_seed=42,
    )

    print(f"\n{'=' * 60}")
    print(f"Running experiment: {label}")
    print(f"{'=' * 60}")

    history = train_agent(
        env=env,
        agent=agent,
        episodes=EPISODES,
        decay_epsilon=decay_epsilon,
        progress_interval=50,
    )
    path, reached_goal = extract_optimal_path(env, agent)
    convergence = convergence_episode(history["successes"], window=20)

    return {
        "label": label,
        "env": env,
        "agent": agent,
        "history": history,
        "path": path,
        "reached_goal": reached_goal,
        "convergence_episode": convergence,
    }


def summarize_result(result: Dict[str, object]) -> None:
    """Print the useful final outputs for one experiment."""
    label = result["label"]
    env = result["env"]
    agent = result["agent"]
    history = result["history"]
    path = result["path"]
    reached_goal = result["reached_goal"]
    convergence = result["convergence_episode"]

    average_reward = float(np.mean(history["rewards"][-50:]))
    average_steps = float(np.mean(history["steps"][-50:]))

    print(f"\nSummary for {label}")
    print(f"Reached goal with greedy policy: {reached_goal}")
    print(f"Average reward over last 50 episodes: {average_reward:.2f}")
    print(f"Average steps over last 50 episodes: {average_steps:.2f}")
    print(
        "Convergence episode: "
        + (str(convergence) if convergence is not None else "Not reached within the chosen window")
    )
    print(f"Learned path: {path}")
    print("\nPath visualization:")
    print(env.render_path(path))
    print("\nFinal Q-table:")
    print_q_table(agent)


def create_plots(fixed_result: Dict[str, object], decaying_result: Dict[str, object]) -> None:
    """Generate and save experiment plots."""
    fixed_history = fixed_result["history"]
    decaying_history = decaying_result["history"]

    reward_plot_path = output_path(PROJECT_DIR, "reward_vs_episode.png")
    steps_plot_path = output_path(PROJECT_DIR, "steps_vs_episode.png")
    epsilon_plot_path = output_path(PROJECT_DIR, "epsilon_decay.png")

    plot_comparison_metric(
        fixed_values=fixed_history["rewards"],
        decaying_values=decaying_history["rewards"],
        ylabel="Total Reward",
        title="Reward vs Episode",
        save_path=reward_plot_path,
    )
    plot_comparison_metric(
        fixed_values=fixed_history["steps"],
        decaying_values=decaying_history["steps"],
        ylabel="Steps",
        title="Steps vs Episode",
        save_path=steps_plot_path,
    )
    plot_epsilon_decay(decaying_history["epsilons"], epsilon_plot_path)

    print("\nSaved plots:")
    print(reward_plot_path)
    print(steps_plot_path)
    print(epsilon_plot_path)


def print_comparison(fixed_result: Dict[str, object], decaying_result: Dict[str, object]) -> None:
    """Print a short comparison of both experiments."""
    fixed_convergence = fixed_result["convergence_episode"]
    decay_convergence = decaying_result["convergence_episode"]

    print("\n" + "=" * 60)
    print("Experiment Comparison")
    print("=" * 60)

    if fixed_convergence is None and decay_convergence is None:
        print("Neither experiment met the chosen convergence criterion within the training horizon.")
    elif fixed_convergence is None:
        print(f"Decaying epsilon converged first at episode {decay_convergence}.")
    elif decay_convergence is None:
        print(f"Fixed epsilon converged first at episode {fixed_convergence}.")
    elif decay_convergence < fixed_convergence:
        print(
            f"Decaying epsilon converged faster (episode {decay_convergence}) "
            f"than fixed epsilon (episode {fixed_convergence})."
        )
    elif fixed_convergence < decay_convergence:
        print(
            f"Fixed epsilon converged faster (episode {fixed_convergence}) "
            f"than decaying epsilon (episode {decay_convergence})."
        )
    else:
        print(f"Both experiments converged at the same episode: {fixed_convergence}.")


def main() -> None:
    fixed_result = run_experiment(label="Fixed epsilon = 0.1", epsilon=0.1, decay_epsilon=False)
    decaying_result = run_experiment(label="Decaying epsilon from 1.0", epsilon=1.0, decay_epsilon=True)

    summarize_result(fixed_result)
    summarize_result(decaying_result)
    print_comparison(fixed_result, decaying_result)
    create_plots(fixed_result, decaying_result)


if __name__ == "__main__":
    main()
