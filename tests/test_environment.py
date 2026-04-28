import pytest
import numpy as np
from src.simulation.vehicle import Vehicle, Direction, VehicleState
from src.simulation.traffic_light import TrafficLight, Phase
from src.simulation.intersection import Intersection
from src.simulation.environment import TrafficEnv
from src.utils.config import load_config


def test_vehicle_movement():
    """Test that a vehicle accelerates and moves correctly."""
    v = Vehicle(x=400, y=800, direction=Direction.NORTH, max_speed=3.0)
    assert v.speed == 3.0
    
    initial_y = v.y
    v.update()
    
    # Should move North (decreasing y)
    assert v.y < initial_y
    assert v.speed == 3.0


def test_traffic_light_transitions():
    """Test the phase cycling of the traffic light."""
    tl = TrafficLight(green_duration=1.0, yellow_duration=1.0, all_red_duration=1.0, fps=10)
    
    # Initial state
    assert tl.current_phase == Phase.NS_GREEN
    
    # Run through green phase
    for _ in range(10):
        tl.update()
        
    assert tl.current_phase == Phase.NS_YELLOW
    
    # Run through yellow phase
    for _ in range(10):
        tl.update()
        
    assert tl.current_phase == Phase.ALL_RED_1


def test_environment_reset_and_step():
    """Test the Gym environment wrapper returns valid observations."""
    env = TrafficEnv(max_steps=100)
    obs, info = env.reset()
    
    # Check observation shape
    assert obs.shape == (22,)
    assert isinstance(obs, np.ndarray)
    
    # Check step behavior
    next_obs, reward, terminated, truncated, info = env.step(action=0)
    
    assert next_obs.shape == (22,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "avg_wait_time" in info
