"""
Gymnasium Environment for the Urban Digital Twin.

Wraps the Intersection simulation as a standard Gymnasium environment
so any RL algorithm (DQN, PPO, etc.) can plug in via env.step(action).

Observation: 14-dim vector (queue lengths, phase one-hot, elapsed time, densities)
Action: Discrete(3) — keep current, switch to NS green, switch to EW green
Reward: Combination of wait-time penalty, throughput bonus, and switching penalty
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .intersection import Intersection, Direction, Phase, Weather
from .vehicle import Direction, VehicleState
from .traffic_light import Phase


class TrafficEnv(gym.Env):
    """
    OpenAI Gymnasium environment for RL-based traffic light control.

    State Space (14 continuous values):
        [0-3]  Queue lengths (N, S, E, W) — normalized by max_queue
        [4-7]  One-hot current phase (NS_GREEN, NS_YELLOW/ALL_RED, EW_GREEN, EW_YELLOW/ALL_RED)
        [8]    Phase elapsed time — normalized by max green duration
        [9-12] Approaching vehicle density (N, S, E, W) — normalized
        [13]   Total waiting vehicles — normalized

    Action Space (Discrete 3):
        0: Keep current phase (do nothing)
        1: Request NS Green
        2: Request EW Green

    Reward:
        - Negative avg wait time (primary signal)
        - Throughput bonus
        - Phase switching penalty (discourages rapid toggling)
    """

    metadata = {"render_modes": ["human", None]}

    def __init__(self, config=None, render_mode=None, max_steps=3600):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps  # 3600 steps = 60 seconds at 60fps... or 1 hour sim time
        self.current_step = 0

        # Default config values
        sim_size = 800
        lane_width = 50
        max_speed = 3.0
        vehicle_length = 40
        vehicle_width = 24
        min_gap = 10
        spawn_rate = 0.02
        emergency_spawn_rate = 0.001
        rush_hour_multiplier = 3.0
        green_dur = 20.0
        yellow_dur = 3.0
        allred_dur = 2.0
        fps = 60

        if config is not None:
            sim = config.simulation
            sim_size = sim.sim_area_size
            lane_width = sim.road.lane_width
            max_speed = sim.vehicle.max_speed
            vehicle_length = sim.vehicle.length
            vehicle_width = sim.vehicle.width
            min_gap = sim.vehicle.min_gap
            spawn_rate = sim.vehicle.spawn_rate
            emergency_spawn_rate = getattr(sim.vehicle, "emergency_spawn_rate", 0.001)
            rush_hour_multiplier = getattr(sim.vehicle, "rush_hour_multiplier", 3.0)
            green_dur = sim.traffic_light.green_duration
            yellow_dur = sim.traffic_light.yellow_duration
            allred_dur = sim.traffic_light.all_red_duration
            fps = sim.fps

        self.fps = fps
        self.green_duration_frames = int(green_dur * fps)

        # Store params for intersection creation
        self._intersection_params = dict(
            sim_size=sim_size, lane_width=lane_width, max_speed=max_speed,
            vehicle_length=vehicle_length, vehicle_width=vehicle_width,
            min_gap=min_gap, spawn_rate=spawn_rate, emergency_spawn_rate=emergency_spawn_rate,
            green_duration=green_dur, yellow_duration=yellow_dur,
            all_red_duration=allred_dur, fps=fps,
            rush_hour_multiplier=rush_hour_multiplier, max_steps=self.max_steps
        )

        self.intersection = Intersection(**self._intersection_params)

        # Renderer (only if human mode)
        self.renderer = None
        if render_mode == "human":
            from .renderer import Renderer
            sidebar = 400
            self.renderer = Renderer(sim_size=sim_size, sidebar_width=sidebar, fps=fps)

        # --- Spaces ---
        self.max_queue = 30  # Normalization cap
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(22,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # keep, NS_green, EW_green

        # Decision frequency: agent acts every N simulation steps
        # (not every single frame — gives time for phase transitions)
        self.decision_interval = int(5 * fps)  # Every 5 seconds
        self._steps_since_decision = 0

        # Track previous metrics for reward shaping
        self._prev_total_wait = 0
        self._prev_cleared = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.intersection = Intersection(**self._intersection_params)
        self.current_step = 0
        self._steps_since_decision = 0
        self._prev_total_wait = 0
        self._prev_cleared = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        """
        Execute one agent decision, then simulate for decision_interval frames.
        """
        # Apply action
        self._apply_action(action)

        # Simulate for decision_interval steps
        total_reward = 0.0
        for _ in range(self.decision_interval):
            self.intersection.step()
            self.current_step += 1

            if self.render_mode == "human" and self.renderer:
                if not self.renderer.render(self.intersection):
                    return self._get_obs(), 0.0, True, False, self._get_info()

        # Compute reward over the interval
        reward = self._compute_reward()
        total_reward += reward

        # Check termination
        terminated = False
        truncated = self.current_step >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _apply_action(self, action):
        """Convert agent action to traffic light command."""
        tl = self.intersection.traffic_light

        if action == 0:
            # Keep current phase — do nothing
            pass
        elif action == 1:
            # Request NS Green
            if tl.current_phase != Phase.NS_GREEN:
                tl.set_phase(Phase.NS_GREEN)
        elif action == 2:
            # Request EW Green
            if tl.current_phase != Phase.EW_GREEN:
                tl.set_phase(Phase.EW_GREEN)

    def _get_obs(self):
        """Build the 14-dim observation vector."""
        inter = self.intersection
        tl = inter.traffic_light

        # Queue lengths (normalized)
        queues = inter.queue_lengths
        q_north = min(queues["NORTH"] / self.max_queue, 1.0)
        q_south = min(queues["SOUTH"] / self.max_queue, 1.0)
        q_east = min(queues["EAST"] / self.max_queue, 1.0)
        q_west = min(queues["WEST"] / self.max_queue, 1.0)

        # Phase one-hot (4 categories: NS_GREEN, transition, EW_GREEN, transition)
        phase_vec = [0.0, 0.0, 0.0, 0.0]
        if tl.current_phase == Phase.NS_GREEN:
            phase_vec[0] = 1.0
        elif tl.current_phase in (Phase.NS_YELLOW, Phase.ALL_RED_1):
            phase_vec[1] = 1.0
        elif tl.current_phase == Phase.EW_GREEN:
            phase_vec[2] = 1.0
        else:  # EW_YELLOW, ALL_RED_2
            phase_vec[3] = 1.0

        # Phase elapsed (normalized by green duration)
        elapsed_norm = min(tl.phase_elapsed / max(self.green_duration_frames, 1), 1.0)

        # Approaching density (vehicles not yet at stop line, normalized)
        densities = []
        for d in [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]:
            approaching = sum(1 for v in inter.vehicles[d]
                              if v.state == VehicleState.APPROACHING)
            densities.append(min(approaching / 10.0, 1.0))

        # Total waiting (normalized)
        total_wait_norm = min(inter.total_waiting / (self.max_queue * 4), 1.0)

        # Emergency vehicle flags
        emergencies = []
        for d in [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]:
            has_emergency = any(v.is_emergency for v in inter.vehicles[d])
            emergencies.append(1.0 if has_emergency else 0.0)

        # Time of day (normalized)
        tod_norm = inter.time_of_day_fraction

        # Weather one-hot
        weather_vec = [
            1.0 if inter.current_weather == Weather.CLEAR else 0.0,
            1.0 if inter.current_weather == Weather.RAIN else 0.0,
            1.0 if inter.current_weather == Weather.SNOW else 0.0,
        ]

        obs = np.array([
            q_north, q_south, q_east, q_west,
            *phase_vec,
            elapsed_norm,
            *densities,
            total_wait_norm,
            *emergencies,
            tod_norm,
            *weather_vec,
        ], dtype=np.float32)

        return obs

    def _compute_reward(self):
        """
        Reward function: minimize wait times, maximize throughput.

        Components:
        1. Wait time penalty (primary): negative of avg wait time change
        2. Throughput bonus: vehicles cleared this interval
        3. Switching penalty: discourage rapid phase toggling
        """
        inter = self.intersection

        # 1. Wait time penalty
        current_total_wait = sum(
            v.wait_time for d in Direction
            for v in inter.vehicles[d]
        )
        wait_delta = current_total_wait - self._prev_total_wait
        wait_penalty = -wait_delta / (self.fps * 10)  # Normalize

        # 2. Throughput bonus
        cleared_delta = inter.total_vehicles_cleared - self._prev_cleared
        throughput_bonus = cleared_delta * 0.3

        # 3. Queue balance penalty (penalize very uneven queues)
        queues = list(inter.queue_lengths.values())
        if max(queues) > 0:
            balance_penalty = -0.1 * (max(queues) - min(queues)) / self.max_queue
        else:
            balance_penalty = 0.0

        # 4. Emergency vehicle penalty
        emergency_penalty = 0.0
        for d in Direction:
            for v in inter.vehicles[d]:
                if v.is_emergency and v.state == VehicleState.WAITING:
                    emergency_penalty -= 10.0  # Massive penalty per frame per waiting ambulance

        # Update tracking
        self._prev_total_wait = current_total_wait
        self._prev_cleared = inter.total_vehicles_cleared

        reward = wait_penalty + throughput_bonus + balance_penalty + emergency_penalty

        return float(reward)

    def _get_info(self):
        """Return metrics dict for logging."""
        inter = self.intersection
        return {
            "avg_wait_time": inter.avg_wait_time,
            "total_waiting": inter.total_waiting,
            "total_cleared": inter.total_vehicles_cleared,
            "throughput_per_min": inter.throughput_per_minute,
            "queue_lengths": inter.queue_lengths,
            "sim_time": inter.sim_time_seconds,
            "phase": inter.traffic_light.current_phase.name,
        }

    def close(self):
        if self.renderer:
            self.renderer.close()
