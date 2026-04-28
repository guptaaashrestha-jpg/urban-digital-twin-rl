"""
Traffic Light module for the Urban Digital Twin.

Manages signal phases, timing, and transitions for a single intersection.
Supports fixed-timer cycling and external control (for RL agent in Phase 3).
"""

from enum import Enum, auto
from typing import Dict


class Phase(Enum):
    """Traffic signal phases for a 4-way intersection."""
    NS_GREEN = auto()    # North-South green, East-West red
    NS_YELLOW = auto()   # North-South yellow (transition)
    ALL_RED_1 = auto()   # All red (clearance before EW green)
    EW_GREEN = auto()    # East-West green, North-South red
    EW_YELLOW = auto()   # East-West yellow (transition)
    ALL_RED_2 = auto()   # All red (clearance before NS green)


# Which directions get green in each phase
PHASE_GREEN_DIRS = {
    Phase.NS_GREEN:  {"NORTH", "SOUTH"},
    Phase.NS_YELLOW: set(),  # Yellow = nobody should enter
    Phase.ALL_RED_1: set(),
    Phase.EW_GREEN:  {"EAST", "WEST"},
    Phase.EW_YELLOW: set(),
    Phase.ALL_RED_2: set(),
}

# Phase cycle order
PHASE_ORDER = [
    Phase.NS_GREEN,
    Phase.NS_YELLOW,
    Phase.ALL_RED_1,
    Phase.EW_GREEN,
    Phase.EW_YELLOW,
    Phase.ALL_RED_2,
]


class TrafficLight:
    """
    Traffic light controller for a single intersection.

    Manages phase cycling with configurable durations.
    Can operate in fixed-timer mode or be externally controlled by an RL agent.

    Attributes:
        current_phase: The active signal phase.
        phase_timer: Frames remaining in the current phase.
        phase_elapsed: Frames elapsed in the current phase.
    """

    def __init__(
        self,
        green_duration: float = 20.0,
        yellow_duration: float = 3.0,
        all_red_duration: float = 2.0,
        fps: int = 60,
    ):
        """
        Initialize traffic light with timing in seconds.

        Args:
            green_duration: Seconds for each green phase.
            yellow_duration: Seconds for yellow transition.
            all_red_duration: Seconds for all-red clearance.
            fps: Simulation frames per second (for converting seconds to frames).
        """
        self.fps = fps

        # Store durations in frames
        self.durations: Dict[Phase, int] = {
            Phase.NS_GREEN:  int(green_duration * fps),
            Phase.NS_YELLOW: int(yellow_duration * fps),
            Phase.ALL_RED_1: int(all_red_duration * fps),
            Phase.EW_GREEN:  int(green_duration * fps),
            Phase.EW_YELLOW: int(yellow_duration * fps),
            Phase.ALL_RED_2: int(all_red_duration * fps),
        }

        # Start with NS green
        self.current_phase = Phase.NS_GREEN
        self.phase_index = 0
        self.phase_timer = self.durations[self.current_phase]
        self.phase_elapsed = 0

        # Tracking
        self.total_switches = 0
        self.phase_just_changed = False

    def update(self):
        """Advance the traffic light by one frame (fixed-timer mode)."""
        self.phase_elapsed += 1
        self.phase_timer -= 1
        self.phase_just_changed = False

        if self.phase_timer <= 0:
            self._advance_phase()

    def _advance_phase(self):
        """Move to the next phase in the cycle."""
        self.phase_index = (self.phase_index + 1) % len(PHASE_ORDER)
        self.current_phase = PHASE_ORDER[self.phase_index]
        self.phase_timer = self.durations[self.current_phase]
        self.phase_elapsed = 0
        self.total_switches += 1
        self.phase_just_changed = True

    def set_phase(self, phase: Phase):
        """
        Externally set the phase (for RL agent control).
        Only allows switching to a green phase — yellow/all-red transitions
        are handled automatically.
        """
        if phase in (Phase.NS_GREEN, Phase.EW_GREEN):
            if phase != self.current_phase and self.current_phase in (Phase.NS_GREEN, Phase.EW_GREEN):
                # Need to go through yellow transition
                if self.current_phase == Phase.NS_GREEN:
                    self.current_phase = Phase.NS_YELLOW
                    self.phase_index = PHASE_ORDER.index(Phase.NS_YELLOW)
                else:
                    self.current_phase = Phase.EW_YELLOW
                    self.phase_index = PHASE_ORDER.index(Phase.EW_YELLOW)
                self.phase_timer = self.durations[self.current_phase]
                self.phase_elapsed = 0
                self.total_switches += 1
                self.phase_just_changed = True

    def is_green(self, direction_name: str) -> bool:
        """Check if a given direction currently has green."""
        return direction_name in PHASE_GREEN_DIRS[self.current_phase]

    def is_yellow(self) -> bool:
        """Check if currently in a yellow phase."""
        return self.current_phase in (Phase.NS_YELLOW, Phase.EW_YELLOW)

    def is_all_red(self) -> bool:
        """Check if currently in an all-red clearance phase."""
        return self.current_phase in (Phase.ALL_RED_1, Phase.ALL_RED_2)

    @property
    def ns_color(self) -> str:
        """Signal color for North-South directions."""
        if self.current_phase == Phase.NS_GREEN:
            return "green"
        elif self.current_phase == Phase.NS_YELLOW:
            return "yellow"
        else:
            return "red"

    @property
    def ew_color(self) -> str:
        """Signal color for East-West directions."""
        if self.current_phase == Phase.EW_GREEN:
            return "green"
        elif self.current_phase == Phase.EW_YELLOW:
            return "yellow"
        else:
            return "red"

    @property
    def phase_progress(self) -> float:
        """Progress through current phase (0.0 to 1.0)."""
        duration = self.durations[self.current_phase]
        return self.phase_elapsed / duration if duration > 0 else 1.0

    @property
    def phase_time_remaining_seconds(self) -> float:
        """Seconds remaining in current phase."""
        return self.phase_timer / self.fps

    def __repr__(self):
        return (
            f"TrafficLight(phase={self.current_phase.name}, "
            f"remaining={self.phase_time_remaining_seconds:.1f}s)"
        )
