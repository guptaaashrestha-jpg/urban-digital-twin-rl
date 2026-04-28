"""
Urban Digital Twin - DQN Training Script

Trains the DQN agent to optimize traffic light control.
Logs training curves and saves model checkpoints.

Usage:
    python scripts/train.py
    python scripts/train.py --episodes 200 --render
    python scripts/train.py --config configs/default.yaml
"""

import sys
import os
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.simulation.environment import TrafficEnv
from src.agents.dqn_agent import DQNAgent
from src.agents.fixed_timer import FixedTimerAgent


def run_evaluation(env, agent, num_episodes=10, agent_name="agent"):
    """Run evaluation episodes and return average metrics."""
    total_wait = []
    total_cleared = []
    total_rewards = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(obs, training=False)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_wait.append(info["avg_wait_time"])
        total_cleared.append(info["total_cleared"])
        total_rewards.append(episode_reward)

    return {
        "agent": agent_name,
        "avg_wait_time": np.mean(total_wait),
        "std_wait_time": np.std(total_wait),
        "avg_throughput": np.mean(total_cleared),
        "avg_reward": np.mean(total_rewards),
    }


def main():
    parser = argparse.ArgumentParser(description="Train DQN for traffic control")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=7200,
                        help="Max simulation frames per episode (7200 = 2 min at 60fps)")
    parser.add_argument("--render", action="store_true",
                        help="Render during training (slower)")
    parser.add_argument("--eval-interval", type=int, default=25,
                        help="Evaluate every N episodes")
    parser.add_argument("--save-dir", type=str, default="models")
    args = parser.parse_args()

    config = load_config(args.config)

    print("=" * 60)
    print("  URBAN DIGITAL TWIN - DQN Training")
    print("=" * 60)

    # Create environment
    render_mode = "human" if args.render else None
    env = TrafficEnv(config=config, render_mode=render_mode, max_steps=args.max_steps)

    # Create DQN agent
    agent = DQNAgent(
        state_dim=22,
        action_dim=3,
        hidden_size=config.training.hidden_size,
        lr=config.training.learning_rate,
        gamma=config.training.gamma,
        epsilon_start=config.training.epsilon_start,
        epsilon_end=config.training.epsilon_end,
        epsilon_decay=config.training.epsilon_decay,
        batch_size=config.training.batch_size,
        buffer_size=config.training.replay_buffer_size,
        target_update_tau=config.training.target_update_tau,
        device=config.training.device,
    )

    # -- Training Loop --
    episode_rewards = []
    episode_waits = []
    best_avg_wait = float("inf")
    start_time = time.time()

    print(f"\n  Training for {args.episodes} episodes...")
    print(f"  Decision interval: {env.decision_interval} frames ({env.decision_interval/env.fps:.1f}s)")
    print(f"  Max steps/episode: {args.max_steps}")
    print(f"  Epsilon: {agent.epsilon_start} -> {agent.epsilon_end}")
    print()

    for episode in range(1, args.episodes + 1):
        obs, info = env.reset()
        episode_reward = 0
        episode_losses = []
        done = False
        steps = 0

        while not done:
            # Select action
            action = agent.select_action(obs, training=True)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store and learn
            agent.store_transition(obs, action, reward, next_obs, done)
            loss = agent.learn()
            if loss is not None:
                episode_losses.append(loss)

            obs = next_obs
            episode_reward += reward
            steps += 1

        episode_rewards.append(episode_reward)
        episode_waits.append(info["avg_wait_time"])

        # Progress logging
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        elapsed = time.time() - start_time

        if episode % 5 == 0 or episode == 1:
            print(
                f"  Episode {episode:4d}/{args.episodes} | "
                f"Reward: {episode_reward:7.1f} | "
                f"AvgWait: {info['avg_wait_time']:5.1f}s | "
                f"Cleared: {info['total_cleared']:4d} | "
                f"e: {agent.epsilon:.3f} | "
                f"Loss: {avg_loss:.4f} | "
                f"Time: {elapsed:.0f}s"
            )

        # Periodic evaluation
        if episode % args.eval_interval == 0:
            print(f"\n  -- Evaluation at Episode {episode} --")
            eval_result = run_evaluation(env, agent, num_episodes=5, agent_name="DQN")
            print(f"  DQN  -> AvgWait: {eval_result['avg_wait_time']:.1f}s | "
                  f"Throughput: {eval_result['avg_throughput']:.0f}")

            # Save best model
            if eval_result["avg_wait_time"] < best_avg_wait:
                best_avg_wait = eval_result["avg_wait_time"]
                save_path = os.path.join(args.save_dir, "best_dqn.pt")
                os.makedirs(args.save_dir, exist_ok=True)
                agent.save(save_path)
                print(f"  * New best! Saved to {save_path}")
            print()

        # Save periodic checkpoints
        if episode % 100 == 0:
            save_path = os.path.join(args.save_dir, f"dqn_ep{episode}.pt")
            os.makedirs(args.save_dir, exist_ok=True)
            agent.save(save_path)

    # -- Final Comparison --
    print("\n" + "=" * 60)
    print("  FINAL COMPARISON: DQN vs Fixed Timer")
    print("=" * 60)

    # Evaluate trained DQN
    dqn_result = run_evaluation(env, agent, num_episodes=20, agent_name="DQN")

    # Evaluate fixed timer baseline
    baseline = FixedTimerAgent()
    baseline_result = run_evaluation(env, baseline, num_episodes=20, agent_name="Fixed Timer")

    # Print comparison
    print(f"\n  {'Metric':<25} {'DQN':>12} {'Fixed Timer':>12} {'Improvement':>12}")
    print(f"  {'-'*61}")

    dqn_wait = dqn_result["avg_wait_time"]
    fix_wait = baseline_result["avg_wait_time"]
    wait_improvement = ((fix_wait - dqn_wait) / fix_wait * 100) if fix_wait > 0 else 0

    print(f"  {'Avg Wait Time (s)':<25} {dqn_wait:>11.1f}s {fix_wait:>11.1f}s {wait_improvement:>+11.1f}%")
    print(f"  {'Avg Throughput':<25} {dqn_result['avg_throughput']:>12.0f} "
          f"{baseline_result['avg_throughput']:>12.0f}")
    print(f"  {'Avg Episode Reward':<25} {dqn_result['avg_reward']:>12.1f} "
          f"{baseline_result['avg_reward']:>12.1f}")

    # Save final model
    final_path = os.path.join(args.save_dir, "final_dqn.pt")
    os.makedirs(args.save_dir, exist_ok=True)
    agent.save(final_path)

    # Save training curves
    _save_training_curves(episode_rewards, episode_waits, agent, args.save_dir)

    elapsed = time.time() - start_time
    print(f"\n  Total training time: {elapsed/60:.1f} minutes")
    print("=" * 60)

    env.close()


def _save_training_curves(rewards, waits, agent, save_dir):
    """Plot and save training curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Urban Digital Twin - DQN Training Results",
                     fontsize=16, fontweight="bold")

        # Reward curve
        ax = axes[0, 0]
        ax.plot(rewards, alpha=0.3, color="#3b82f6")
        if len(rewards) >= 20:
            smooth = np.convolve(rewards, np.ones(20)/20, mode="valid")
            ax.plot(range(19, len(rewards)), smooth, color="#3b82f6", linewidth=2)
        ax.set_title("Episode Reward")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.grid(alpha=0.3)

        # Wait time curve
        ax = axes[0, 1]
        ax.plot(waits, alpha=0.3, color="#ef4444")
        if len(waits) >= 20:
            smooth = np.convolve(waits, np.ones(20)/20, mode="valid")
            ax.plot(range(19, len(waits)), smooth, color="#ef4444", linewidth=2)
        ax.set_title("Average Wait Time")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Seconds")
        ax.grid(alpha=0.3)

        # Loss curve
        ax = axes[1, 0]
        losses = agent.training_losses
        if losses:
            # Downsample if too many points
            step = max(1, len(losses) // 1000)
            sampled = losses[::step]
            ax.plot(sampled, alpha=0.3, color="#a855f7")
            if len(sampled) >= 50:
                smooth = np.convolve(sampled, np.ones(50)/50, mode="valid")
                ax.plot(range(49, len(sampled)), smooth, color="#a855f7", linewidth=2)
        ax.set_title("Training Loss")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Huber Loss")
        ax.grid(alpha=0.3)

        # Epsilon decay
        ax = axes[1, 1]
        eps_vals = [agent.epsilon_end + (agent.epsilon_start - agent.epsilon_end) *
                    np.exp(-i / agent.epsilon_decay) for i in range(agent.steps_done + 1)]
        step = max(1, len(eps_vals) // 500)
        ax.plot(eps_vals[::step], color="#22c55e", linewidth=2)
        ax.set_title("Exploration Rate (epsilon)")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Epsilon")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(save_dir, "training_curves.png")
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Training curves saved to {plot_path}")
    except Exception as e:
        print(f"  Warning: Could not save training curves: {e}")


if __name__ == "__main__":
    main()
