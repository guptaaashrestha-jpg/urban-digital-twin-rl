"""
Urban Digital Twin — Evaluate & Log Telemetry

Loads the trained DQN model and compares it against the Fixed-Timer baseline
over multiple episodes. Renders the simulation to the screen and logs all
telemetry to SQLite, then exports to CSV for the Power BI dashboard.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --model models/best_dqn.pt --episodes 10
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.simulation.environment import TrafficEnv
from src.agents.dqn_agent import DQNAgent
from src.agents.fixed_timer import FixedTimerAgent
from src.data.telemetry_logger import TelemetryLogger


def evaluate_agent(env, agent, num_episodes, logger, agent_name):
    """Run an agent for N episodes and log all telemetry."""
    print(f"\n  Evaluating {agent_name}...")
    
    total_wait_times = []
    total_throughputs = []
    
    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        episode_id = logger.start_episode(agent_name)
        
        episode_reward = 0
        done = False
        step = 0
        
        print(f"    Running Episode {ep}/{num_episodes}...", end="", flush=True)
        
        while not done:
            action = agent.select_action(obs, training=False)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            step += 1
            
            # Log step telemetry
            logger.log_step(episode_id, step, action, reward, info)
            
        # End episode
        logger.end_episode(
            episode_id, 
            total_steps=step, 
            total_reward=episode_reward, 
            avg_wait=info["avg_wait_time"], 
            throughput=info["total_cleared"]
        )
        
        total_wait_times.append(info["avg_wait_time"])
        total_throughputs.append(info["total_cleared"])
        print(f" Done (Avg Wait: {info['avg_wait_time']:.1f}s, Throughput: {info['total_cleared']})")
        
    return np.mean(total_wait_times), np.mean(total_throughputs)


def main():
    parser = argparse.ArgumentParser(description="Evaluate agents and log telemetry")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default="models/best_dqn.pt", help="Path to trained DQN model")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per agent")
    parser.add_argument("--max-steps", type=int, default=3600, help="Steps per episode")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering for faster evaluation")
    args = parser.parse_args()

    config = load_config(args.config)
    
    print("=" * 60)
    print("  URBAN DIGITAL TWIN - Evaluation & Telemetry Logging")
    print("=" * 60)

    # Initialize Environment
    render_mode = None if args.no_render else "human"
    env = TrafficEnv(config=config, render_mode=render_mode, max_steps=args.max_steps)
    
    # Initialize Logger
    logger = TelemetryLogger(db_path=config.telemetry.db_path)
    
    # Evaluate Fixed Timer Baseline
    baseline = FixedTimerAgent()
    baseline_wait, baseline_throughput = evaluate_agent(
        env, baseline, args.episodes, logger, "fixed_timer"
    )
    
    # Evaluate Trained DQN
    dqn = DQNAgent(
        state_dim=22,
        action_dim=3,
        hidden_size=config.training.hidden_size,
        device=config.training.device
    )
    
    if os.path.exists(args.model):
        dqn.load(args.model)
        dqn_wait, dqn_throughput = evaluate_agent(
            env, dqn, args.episodes, logger, "dqn"
        )
    else:
        print(f"\n  [ERROR] Model file not found: {args.model}")
        print("  Please train the model first by running `python scripts/train.py`")
        return

    # Export to CSV for Power BI
    print("\n  Exporting telemetry data...")
    logger.export_to_csv(config.telemetry.export_dir)
    logger.close()
    env.close()
    
    # Summary
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  {'Metric':<25} {'DQN':>12} {'Fixed Timer':>12} {'Improvement':>12}")
    print(f"  {'-'*61}")
    
    wait_improvement = ((baseline_wait - dqn_wait) / baseline_wait * 100) if baseline_wait > 0 else 0
    throughput_improvement = ((dqn_throughput - baseline_throughput) / baseline_throughput * 100) if baseline_throughput > 0 else 0
    
    print(f"  {'Avg Wait Time (s)':<25} {dqn_wait:>11.1f}s {baseline_wait:>11.1f}s {wait_improvement:>+11.1f}%")
    print(f"  {'Avg Throughput':<25} {dqn_throughput:>12.0f} {baseline_throughput:>12.0f} {throughput_improvement:>+11.1f}%")
    print("=" * 60)
    print(f"  Data ready for Power BI at: {config.telemetry.export_dir}")


if __name__ == "__main__":
    main()
