"""
Vehicle module for the Urban Digital Twin.

Defines the Vehicle entity with movement, queuing, and wait-time tracking.
Vehicles spawn at screen edges, travel toward the intersection, and stop
at red lights or behind other queued vehicles.
"""

import random
from enum import Enum, auto
from typing import Tuple


class Direction(Enum):
    """Cardinal direction a vehicle is traveling TOWARD."""
    NORTH = auto()  # Moving upward (y decreasing), spawns at bottom
    SOUTH = auto()  # Moving downward (y increasing), spawns at top
    EAST = auto()   # Moving right (x increasing), spawns at left
    WEST = auto()   # Moving left (x decreasing), spawns at right


class VehicleState(Enum):
    APPROACHING = auto()   # Moving toward intersection
    WAITING = auto()       # Stopped at red light / behind queue
    CROSSING = auto()      # Inside intersection, passing through
    CLEARED = auto()       # Past intersection, will be removed


# --- Vehicle color palette (wait-time gradient) ---
# Fresh vehicle (no wait)
COLOR_FRESH = (72, 199, 142)       # Mint green
# Short wait (< 5s)
COLOR_SHORT_WAIT = (251, 191, 36)  # Amber
# Long wait (> 15s)
COLOR_LONG_WAIT = (239, 68, 68)    # Red

# Vehicle body colors (random selection for visual variety)
VEHICLE_BODY_COLORS = [
    (59, 130, 246),   # Blue
    (168, 85, 247),   # Purple
    (236, 72, 153),   # Pink
    (34, 197, 94),    # Green
    (251, 146, 60),   # Orange
    (255, 255, 255),  # White
    (148, 163, 184),  # Silver
    (250, 204, 21),   # Yellow
]


def _lerp_color(c1: Tuple[int, ...], c2: Tuple[int, ...], t: float) -> Tuple[int, ...]:
    """Linearly interpolate between two RGB colors."""
    t = max(0.0, min(1.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


class Vehicle:
    """
    A single vehicle in the traffic simulation.

    Attributes:
        x, y: Center position in pixels.
        direction: Which way the vehicle is traveling.
        speed: Current speed in pixels/frame.
        max_speed: Maximum speed.
        wait_time: Cumulative frames spent waiting (speed == 0).
        state: Current vehicle state.
    """

    _id_counter = 0

    def __init__(
        self,
        x: float,
        y: float,
        direction: Direction,
        max_speed: float = 3.0,
        length: int = 40,
        width: int = 24,
    ):
        Vehicle._id_counter += 1
        self.id = Vehicle._id_counter

        self.x = float(x)
        self.y = float(y)
        self.direction = direction

        self.max_speed = max_speed
        self.speed = max_speed  # Start at full speed
        self.acceleration = 0.3
        self.deceleration = 0.5

        # Dimensions — length is along travel direction, width across
        self.length = length
        self.width = width

        self.wait_time = 0         # Frames spent at speed 0
        self.total_travel_time = 0  # Total frames alive
        self.state = VehicleState.APPROACHING

        # Random body color for visual variety
        self.body_color = random.choice(VEHICLE_BODY_COLORS)

    # --- Geometry helpers ---

    @property
    def is_vertical(self) -> bool:
        return self.direction in (Direction.NORTH, Direction.SOUTH)

    @property
    def render_width(self) -> int:
        """Width on screen (x-axis)."""
        return self.width if self.is_vertical else self.length

    @property
    def render_height(self) -> int:
        """Height on screen (y-axis)."""
        return self.length if self.is_vertical else self.width

    @property
    def front_pos(self) -> float:
        """Position of the front edge along the axis of travel."""
        if self.direction == Direction.NORTH:
            return self.y - self.length / 2   # Front is top (min y)
        elif self.direction == Direction.SOUTH:
            return self.y + self.length / 2   # Front is bottom (max y)
        elif self.direction == Direction.EAST:
            return self.x + self.length / 2   # Front is right (max x)
        else:  # WEST
            return self.x - self.length / 2   # Front is left (min x)

    @property
    def back_pos(self) -> float:
        """Position of the back edge along the axis of travel."""
        if self.direction == Direction.NORTH:
            return self.y + self.length / 2
        elif self.direction == Direction.SOUTH:
            return self.y - self.length / 2
        elif self.direction == Direction.EAST:
            return self.x - self.length / 2
        else:  # WEST
            return self.x + self.length / 2

    # --- Movement ---

    def update(self, stop_position: float = None, vehicle_ahead_back: float = None, min_gap: float = 10):
        """
        Update vehicle position for one simulation tick.

        Args:
            stop_position: If set, the coordinate where the front must stop
                           (e.g., a red-light stop line).
            vehicle_ahead_back: Back-edge position of the vehicle directly ahead
                                in the queue (for maintaining gap).
            min_gap: Minimum pixel gap to maintain behind vehicle ahead.
        """
        self.total_travel_time += 1

        if self.state == VehicleState.CLEARED:
            # Keep moving out of frame
            self._move_forward()
            return

        # Determine the effective stop point
        effective_stop = None

        if vehicle_ahead_back is not None:
            # Stop behind the vehicle ahead
            if self.direction == Direction.NORTH:
                effective_stop = vehicle_ahead_back + min_gap + self.length / 2
            elif self.direction == Direction.SOUTH:
                effective_stop = vehicle_ahead_back - min_gap - self.length / 2
            elif self.direction == Direction.EAST:
                effective_stop = vehicle_ahead_back - min_gap - self.length / 2
            else:  # WEST
                effective_stop = vehicle_ahead_back + min_gap + self.length / 2

        if stop_position is not None:
            # Stop at the red light stop line
            if self.direction == Direction.NORTH:
                light_stop = stop_position + self.length / 2
                effective_stop = max(light_stop, effective_stop) if effective_stop else light_stop
            elif self.direction == Direction.SOUTH:
                light_stop = stop_position - self.length / 2
                effective_stop = min(light_stop, effective_stop) if effective_stop else light_stop
            elif self.direction == Direction.EAST:
                light_stop = stop_position - self.length / 2
                effective_stop = min(light_stop, effective_stop) if effective_stop else light_stop
            else:  # WEST
                light_stop = stop_position + self.length / 2
                effective_stop = max(light_stop, effective_stop) if effective_stop else light_stop

        if effective_stop is not None:
            # Check if we need to stop or slow down
            if self._should_stop(effective_stop):
                self._decelerate_to_stop(effective_stop)
            else:
                self._accelerate()
        else:
            # No obstacles — full speed
            self._accelerate()

        self._move_forward()

        # Track waiting
        if self.speed < 0.1:
            self.wait_time += 1
            self.state = VehicleState.WAITING
        elif self.state == VehicleState.WAITING:
            self.state = VehicleState.APPROACHING

    def _should_stop(self, stop_at: float) -> bool:
        """Check if we're close enough to the stop point to begin braking."""
        dist = self._distance_to(stop_at)
        # Braking distance: v^2 / (2 * deceleration)
        brake_dist = (self.speed ** 2) / (2 * self.deceleration + 0.01)
        return dist <= brake_dist + 5  # Small buffer

    def _distance_to(self, stop_at: float) -> float:
        """Distance from front of vehicle to the stop point."""
        if self.direction == Direction.NORTH:
            return self.y - self.length / 2 - stop_at  # stop_at > self.front when ahead
            # Actually: front = y - length/2, stop is at larger y, so dist = stop_at - front? No.
            # North = y decreasing. Front = y - l/2. Stop is at a y-value we shouldn't go below.
            # Distance = front_y - stop_y = (y - l/2) - stop_at... but stop_at > front when we haven't reached it
            # Hmm, for north: vehicle is above (lower y) the stop or below (higher y).
            # Vehicle starts at bottom (high y), moves up (y decreases). Front = y - l/2.
            # Stop at some y value. Distance = front - stop = (y - l/2) - stop_at
            # When vehicle hasn't reached stop: front > stop (front is at higher y), dist > 0
            return (self.y - self.length / 2) - stop_at
        elif self.direction == Direction.SOUTH:
            return stop_at - (self.y + self.length / 2)
        elif self.direction == Direction.EAST:
            return stop_at - (self.x + self.length / 2)
        else:  # WEST
            return (self.x - self.length / 2) - stop_at

    def _decelerate_to_stop(self, stop_at: float):
        """Smoothly decelerate to stop at the given position."""
        dist = self._distance_to(stop_at)
        if dist <= 0:
            self.speed = 0
            # Snap to stop position
            if self.direction == Direction.NORTH:
                self.y = stop_at + self.length / 2
            elif self.direction == Direction.SOUTH:
                self.y = stop_at - self.length / 2
            elif self.direction == Direction.EAST:
                self.x = stop_at - self.length / 2
            else:
                self.x = stop_at + self.length / 2
        else:
            # Decelerate proportionally to distance
            target_speed = max(0.5, self.max_speed * (dist / 80))
            self.speed = max(0, min(self.speed, target_speed))

    def _accelerate(self):
        """Accelerate toward max speed."""
        if self.speed < self.max_speed:
            self.speed = min(self.max_speed, self.speed + self.acceleration)

    def _move_forward(self):
        """Move the vehicle in its travel direction."""
        if self.direction == Direction.NORTH:
            self.y -= self.speed
        elif self.direction == Direction.SOUTH:
            self.y += self.speed
        elif self.direction == Direction.EAST:
            self.x += self.speed
        else:  # WEST
            self.x -= self.speed

    # --- Visual ---

    def get_wait_color(self) -> Tuple[int, int, int]:
        """Get color based on wait time (green → yellow → red gradient)."""
        wait_seconds = self.wait_time / 60  # Convert frames to seconds at 60fps
        if wait_seconds < 5:
            t = wait_seconds / 5
            return _lerp_color(COLOR_FRESH, COLOR_SHORT_WAIT, t)
        else:
            t = min((wait_seconds - 5) / 15, 1.0)
            return _lerp_color(COLOR_SHORT_WAIT, COLOR_LONG_WAIT, t)

    def is_off_screen(self, sim_size: int) -> bool:
        """Check if vehicle has left the simulation area."""
        margin = 50
        return (
            self.x < -margin
            or self.x > sim_size + margin
            or self.y < -margin
            or self.y > sim_size + margin
        )

    def __repr__(self):
        return (
            f"Vehicle(id={self.id}, dir={self.direction.name}, "
            f"pos=({self.x:.0f},{self.y:.0f}), spd={self.speed:.1f}, "
            f"wait={self.wait_time})"
        )
