"""
Fixed-Timer baseline agent for comparison.

This agent doesn't make any decisions — it lets the traffic light
run on its default fixed cycle. Used as a baseline to measure
how much the RL agent improves over traditional timing.
"""


class FixedTimerAgent:
    """
    Baseline agent that always returns action 0 (keep current phase).
    The traffic light handles its own fixed-cycle timing.
    """

    def __init__(self):
        self.name = "fixed_timer"

    def select_action(self, state, training=False):
        """Always keep the current phase — let the fixed timer run."""
        return 0

    def store_transition(self, *args):
        pass

    def learn(self):
        return None
