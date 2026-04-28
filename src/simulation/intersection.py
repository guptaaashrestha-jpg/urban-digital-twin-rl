"""
Intersection module for the Urban Digital Twin.

Manages a single 4-way intersection: vehicle spawning, queue management,
movement updates, and throughput tracking.
"""

import random
from typing import List, Dict
from .traffic_light import TrafficLight, Phase
from .vehicle import Vehicle, Direction, VehicleState, Weather


class Intersection:
    """
    A 4-way intersection with traffic light control.

    Right-hand traffic layout (top-down):
    - Northbound: east lane of vertical road
    - Southbound: west lane of vertical road
    - Eastbound: south lane of horizontal road
    - Westbound: north lane of horizontal road
    """

    def __init__(self, sim_size=800, lane_width=50, max_speed=3.0,
                 vehicle_length=40, vehicle_width=24, min_gap=10,
                 spawn_rate=0.02, emergency_spawn_rate=0.001, green_duration=20.0, yellow_duration=3.0,
                 all_red_duration=2.0, fps=60, rush_hour_multiplier=3.0, max_steps=3600):
        self.sim_size = sim_size
        self.lane_width = lane_width
        self.road_width = lane_width * 2
        self.center_x = sim_size // 2
        self.center_y = sim_size // 2

        self.max_speed = max_speed
        self.vehicle_length = vehicle_length
        self.vehicle_width = vehicle_width
        self.min_gap = min_gap
        self.spawn_rate = spawn_rate
        self.emergency_spawn_rate = emergency_spawn_rate
        self.rush_hour_multiplier = rush_hour_multiplier
        self.max_steps = max_steps
        self.fps = fps

        # Intersection bounding box
        self.box_left = self.center_x - self.road_width // 2
        self.box_right = self.center_x + self.road_width // 2
        self.box_top = self.center_y - self.road_width // 2
        self.box_bottom = self.center_y + self.road_width // 2

        # Lane center positions
        self.lane_centers = {
            Direction.NORTH: self.center_x + lane_width // 2,
            Direction.SOUTH: self.center_x - lane_width // 2,
            Direction.EAST:  self.center_y + lane_width // 2,
            Direction.WEST:  self.center_y - lane_width // 2,
        }

        # Stop line positions (edge of intersection box)
        self.stop_lines = {
            Direction.NORTH: self.box_bottom,
            Direction.SOUTH: self.box_top,
            Direction.EAST:  self.box_left,
            Direction.WEST:  self.box_right,
        }

        # Traffic light
        self.traffic_light = TrafficLight(
            green_duration=green_duration,
            yellow_duration=yellow_duration,
            all_red_duration=all_red_duration,
            fps=fps,
        )

        # Vehicle storage
        self.vehicles: Dict[Direction, List[Vehicle]] = {d: [] for d in Direction}
        self.cleared_vehicles: List[Vehicle] = []

        # Metrics
        self.total_vehicles_spawned = 0
        self.total_vehicles_cleared = 0
        self.vehicles_cleared_this_step = 0
        self.cumulative_wait_time = 0
        self.sim_time_frames = 0
        self.total_fines_collected = 0
        self.v2x_synced_count = 0
        self.current_weather = Weather.CLEAR

        # Store config for reset
        self._green_dur = green_duration
        self._yellow_dur = yellow_duration
        self._allred_dur = all_red_duration

    def _spawn_position(self, direction):
        lc = self.lane_centers[direction]
        margin = 20
        if direction == Direction.NORTH:
            return (lc, self.sim_size + margin)
        elif direction == Direction.SOUTH:
            return (lc, -margin)
        elif direction == Direction.EAST:
            return (-margin, lc)
        else:
            return (self.sim_size + margin, lc)

    def _is_in_intersection(self, v):
        return (self.box_left <= v.x <= self.box_right
                and self.box_top <= v.y <= self.box_bottom)

    def _is_past_intersection(self, v):
        if v.direction == Direction.NORTH:
            return v.y + v.length / 2 < self.box_top
        elif v.direction == Direction.SOUTH:
            return v.y - v.length / 2 > self.box_bottom
        elif v.direction == Direction.EAST:
            return v.x - v.length / 2 > self.box_right
        else:
            return v.x + v.length / 2 < self.box_left

    @property
    def time_of_day_fraction(self):
        """0.0 to 1.0 representing the progression through the simulation day."""
        return (self.sim_time_frames % self.max_steps) / max(1, self.max_steps)

    @property
    def current_phase_name(self):
        f = self.time_of_day_fraction
        if f < 0.33:
            return "MORNING RUSH"
        elif f < 0.66:
            return "MID-DAY"
        else:
            return "EVENING RUSH"

    def _get_directional_spawn_rate(self, direction):
        """Scale the base spawn rate based on the time of day."""
        f = self.time_of_day_fraction
        base = self.spawn_rate
        mult = self.rush_hour_multiplier
        
        if f < 0.33:
            # Morning: Heavy North -> South
            if direction == Direction.NORTH:
                return base * mult
            else:
                return base * 0.5
        elif f < 0.66:
            # Mid-Day: Balanced
            return base
        else:
            # Evening: Heavy East -> West
            if direction == Direction.EAST:
                return base * mult
            else:
                return base * 0.5

    def step(self):
        """Advance simulation by one frame."""
        self.sim_time_frames += 1
        self.vehicles_cleared_this_step = 0
        
        # Dynamic Weather transitions
        if self.sim_time_frames % 1200 == 0:
            self.current_weather = random.choice(list(Weather))

        # Broadcast V2X signals
        self._broadcast_v2x_signals()

        # --- Emergency Hardware Preemption ---
        ambulance_ns = any(v.is_emergency for d in (Direction.NORTH, Direction.SOUTH) for v in self.vehicles[d])
        ambulance_ew = any(v.is_emergency for d in (Direction.EAST, Direction.WEST) for v in self.vehicles[d])

        target_phase = None
        if ambulance_ns:
            target_phase = Phase.NS_GREEN
        elif ambulance_ew:
            target_phase = Phase.EW_GREEN

        if target_phase is not None:
            self.traffic_light.set_phase(target_phase)
            # Freeze the timer if we are in the target green phase so it doesn't cycle away
            if self.traffic_light.current_phase == target_phase:
                self.traffic_light.phase_timer = max(self.traffic_light.phase_timer, 2)
        # -------------------------------------

        self.traffic_light.update()
        self._spawn_vehicles()
        for direction in Direction:
            self._update_direction(direction)
        self._update_cleared()
        self._update_metrics()

    def _spawn_vehicles(self):
        for direction in Direction:
            rate = self._get_directional_spawn_rate(direction)
            r = random.random()
            if r < rate + self.emergency_spawn_rate:
                is_emergency = r < self.emergency_spawn_rate
                is_rogue = random.random() < 0.02
                x, y = self._spawn_position(direction)
                v = Vehicle(x=x, y=y, direction=direction,
                            max_speed=self.max_speed,
                            length=self.vehicle_length,
                            width=self.vehicle_width,
                            is_emergency=is_emergency,
                            is_rogue=is_rogue)
                self.vehicles[direction].append(v)
                self.total_vehicles_spawned += 1

    def _update_direction(self, direction):
        queue = self.vehicles[direction]
        is_green = self.traffic_light.is_green(direction.name)

        # Sort: closest to intersection first
        if direction == Direction.NORTH:
            queue.sort(key=lambda v: v.y)
        elif direction == Direction.SOUTH:
            queue.sort(key=lambda v: -v.y)
        elif direction == Direction.EAST:
            queue.sort(key=lambda v: -v.x)
        else:
            queue.sort(key=lambda v: v.x)

        ambulance_present = any(v.is_emergency for v in queue)

        to_remove = []
        for i, vehicle in enumerate(queue):
            if vehicle.state == VehicleState.CROSSING:
                if not is_green and vehicle.is_rogue and not vehicle.was_fined and not vehicle.is_emergency:
                    vehicle.was_fined = True
                    self.total_fines_collected += 500

                vehicle.update()
                if self._is_past_intersection(vehicle):
                    vehicle.state = VehicleState.CLEARED
                    self.cleared_vehicles.append(vehicle)
                    to_remove.append(i)
                    self.total_vehicles_cleared += 1
                    self.vehicles_cleared_this_step += 1
                continue

            if ambulance_present and not vehicle.is_emergency:
                vehicle.pulled_over = True
            else:
                vehicle.pulled_over = False

            stop_pos = None
            vehicle_ahead_back = None

            if not is_green:
                stop_pos = self.stop_lines[direction]

            if vehicle.is_rogue and i > 0 and queue[i - 1].is_emergency and queue[i - 1].state == VehicleState.CROSSING:
                stop_pos = None

            if i > 0:
                for j in range(i - 1, -1, -1):
                    ahead = queue[j]
                    if ahead.state == VehicleState.CROSSING:
                        continue
                    if vehicle.is_emergency and ahead.pulled_over:
                        continue
                    vehicle_ahead_back = ahead.back_pos
                    break

            vehicle.update(stop_position=stop_pos,
                           vehicle_ahead_back=vehicle_ahead_back,
                           min_gap=self.min_gap,
                           weather=self.current_weather)

            if (is_green or stop_pos is None) and self._is_in_intersection(vehicle):
                vehicle.state = VehicleState.CROSSING

        for i in reversed(to_remove):
            queue.pop(i)

    def _update_cleared(self):
        for v in self.cleared_vehicles:
            v._move_forward()
        self.cleared_vehicles = [
            v for v in self.cleared_vehicles
            if not v.is_off_screen(self.sim_size)
        ]

    def _update_metrics(self):
        for direction in Direction:
            for v in self.vehicles[direction]:
                if v.state == VehicleState.WAITING:
                    self.cumulative_wait_time += 1

    @property
    def queue_lengths(self):
        return {
            d.name: sum(1 for v in self.vehicles[d]
                        if v.state == VehicleState.WAITING)
            for d in Direction
        }

    @property
    def total_waiting(self):
        return sum(self.queue_lengths.values())

    @property
    def total_vehicles(self):
        return sum(len(self.vehicles[d]) for d in Direction) + len(self.cleared_vehicles)

    @property
    def avg_wait_time(self):
        waiting = []
        for d in Direction:
            for v in self.vehicles[d]:
                if v.wait_time > 0:
                    waiting.append(v.wait_time / 60.0)
        return sum(waiting) / len(waiting) if waiting else 0.0

    @property
    def throughput_per_minute(self):
        sim_minutes = self.sim_time_frames / (60 * 60)
        return self.total_vehicles_cleared / sim_minutes if sim_minutes > 0 else 0.0

    @property
    def sim_time_seconds(self):
        return self.sim_time_frames / 60.0

    def get_all_vehicles(self):
        all_v = []
        for d in Direction:
            all_v.extend(self.vehicles[d])
        all_v.extend(self.cleared_vehicles)
        return all_v

    def reset(self):
        self.vehicles = {d: [] for d in Direction}
        self.cleared_vehicles: List[Vehicle] = []
        self.total_vehicles_spawned = 0
        self.total_vehicles_cleared = 0
        self.vehicles_cleared_this_step = 0
        self.cumulative_wait_time = 0
        self.sim_time_frames = 0
        self.total_fines_collected = 0
        self.v2x_synced_count = 0
        self.current_weather = Weather.CLEAR
        self.traffic_light = TrafficLight(
            green_duration=self._green_dur,
            yellow_duration=self._yellow_dur,
            all_red_duration=self._allred_dur,
            fps=self.fps,
        )

    def _broadcast_v2x_signals(self):
        self.v2x_synced_count = 0
        tl = self.traffic_light
        hw = self.road_width / 2

        yellow_dur = tl.durations[Phase.NS_YELLOW] / tl.fps
        all_red_dur = tl.durations[Phase.ALL_RED_1] / tl.fps

        # Time to Green (NS)
        if tl.ns_color == "green":
            ns_time = 0.0
        elif tl.ns_color == "yellow":
            ns_time = tl.phase_time_remaining_seconds + all_red_dur
        else: # red
            if tl.ew_color == "green":
                ns_time = tl.phase_time_remaining_seconds + yellow_dur + all_red_dur
            elif tl.ew_color == "yellow":
                ns_time = tl.phase_time_remaining_seconds + all_red_dur
            else:
                ns_time = tl.phase_time_remaining_seconds

        # Time to Green (EW)
        if tl.ew_color == "green":
            ew_time = 0.0
        elif tl.ew_color == "yellow":
            ew_time = tl.phase_time_remaining_seconds + all_red_dur
        else: # red
            if tl.ns_color == "green":
                ew_time = tl.phase_time_remaining_seconds + yellow_dur + all_red_dur
            elif tl.ns_color == "yellow":
                ew_time = tl.phase_time_remaining_seconds + all_red_dur
            else:
                ew_time = tl.phase_time_remaining_seconds

        for d in Direction:
            is_ns = d in (Direction.NORTH, Direction.SOUTH)
            time_to_green = ns_time if is_ns else ew_time
            
            # Stop position for this direction
            if d == Direction.NORTH: stop_pos = self.center_y - hw
            elif d == Direction.SOUTH: stop_pos = self.center_y + hw
            elif d == Direction.EAST: stop_pos = self.center_x + hw
            else: stop_pos = self.center_x - hw

            for v in self.vehicles[d]:
                if v.state == VehicleState.APPROACHING and v.is_v2x_equipped:
                    if time_to_green > 0.5:
                        dist = v._distance_to(stop_pos)
                        if dist > 80: # Only engage V2X if sufficiently far away (80 pixels)
                            frames_to_green = time_to_green * 60.0
                            v.v2x_target_speed = dist / frames_to_green
                            self.v2x_synced_count += 1
                        else:
                            v.v2x_target_speed = None
                    else:
                        v.v2x_target_speed = None
