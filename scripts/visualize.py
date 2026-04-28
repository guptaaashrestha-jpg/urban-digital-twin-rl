"""
Urban Digital Twin — Traffic Simulation Visualizer

Run this script to launch the Pygame traffic simulation with
a fixed-timer traffic light controller. This is the Phase 1 MVP
that validates the simulation logic before adding RL.

Usage:
    python scripts/visualize.py
    python scripts/visualize.py --config configs/default.yaml
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.simulation.intersection import Intersection
from src.simulation.renderer import Renderer


def main():
    parser = argparse.ArgumentParser(description="Urban Digital Twin Visualizer")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    sim = config.simulation

    # Create intersection
    intersection = Intersection(
        sim_size=sim.sim_area_size,
        lane_width=sim.road.lane_width,
        max_speed=sim.vehicle.max_speed,
        vehicle_length=sim.vehicle.length,
        vehicle_width=sim.vehicle.width,
        min_gap=sim.vehicle.min_gap,
        spawn_rate=sim.vehicle.spawn_rate,
        green_duration=sim.traffic_light.green_duration,
        yellow_duration=sim.traffic_light.yellow_duration,
        all_red_duration=sim.traffic_light.all_red_duration,
        fps=sim.fps,
    )

    # Create renderer
    renderer = Renderer(
        sim_size=sim.sim_area_size,
        sidebar_width=sim.window_width - sim.sim_area_size,
        fps=sim.fps,
    )

    print("=" * 50)
    print("  URBAN DIGITAL TWIN — Traffic Simulator")
    print("  Mode: Fixed Timer Baseline")
    print(f"  Green: {sim.traffic_light.green_duration}s | "
          f"Yellow: {sim.traffic_light.yellow_duration}s | "
          f"All-Red: {sim.traffic_light.all_red_duration}s")
    print("  Close the window or press Ctrl+C to exit.")
    print("=" * 50)

    # Main simulation loop
    running = True
    try:
        while running:
            intersection.step()
            running = renderer.render(intersection)
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        # Print final stats
        print(f"\n── Final Statistics ──")
        print(f"  Sim Time:        {intersection.sim_time_seconds:.0f}s")
        print(f"  Vehicles Spawned: {intersection.total_vehicles_spawned}")
        print(f"  Vehicles Cleared: {intersection.total_vehicles_cleared}")
        print(f"  Avg Wait Time:    {intersection.avg_wait_time:.1f}s")
        print(f"  Throughput:       {intersection.throughput_per_minute:.0f} vehicles/min")
        renderer.close()


if __name__ == "__main__":
    main()
