from gymnasium.vector import AsyncVectorEnv
from .drone_env import DroneEnv

def make_drone_env(rank: int, seed: int = 0):
    """Return a function that creates a DroneEnv environment."""
    def _init():
        env = DroneEnv(render_mode=None)
        env.reset(seed=seed + rank)
        return env
    return _init

def create_vector_env(num_envs: int, seed: int = 0):
    """Create an AsyncVectorEnv with num_envs DroneEnv environments."""
    return AsyncVectorEnv(
        [make_drone_env(rank=i, seed=seed) for i in range(num_envs)]
    )
