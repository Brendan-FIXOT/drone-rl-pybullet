import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os

class DroneEnv(gym.Env):
    # to define the render modes and fps for graphical rendering
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    ASSETS_PATH = os.path.expanduser(
        "~/Desktop/gym-pybullet-drones/gym_pybullet_drones/assets"
    )

    def __init__(self, render_mode=None):
        super().__init__()
        """
        Initialize the Drone Environment.
        Action Space: 4 continuous values between 0 and 1 representing motor thrusts.
        """

        self.render_mode = render_mode
        self.time_step = 1/240.0
        self.max_episode_steps = 1000

        self.action_space = spaces.Box(
            low=np.array([0,0,0,0], dtype=np.float32),
            high=np.array([1,1,1,1], dtype=np.float32),
            dtype=np.float32,
            shape=(4,)
        )

        obs_high = np.array([np.inf]*12, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        self.step_counter = 0
        self.target = np.array([0, 0, 1.0], dtype=np.float32)  # Target position (hover)

        # Connect to PyBullet
        if self.render_mode == "human":
            self.client_id = p.connect(p.GUI)
        else:
            # always use DIRECT for training
            self.client_id = p.connect(p.DIRECT)

        p.setTimeStep(self.time_step, physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client_id)

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.setAdditionalSearchPath(self.ASSETS_PATH, physicsClientId=self.client_id)

        # Initialize the simulation
        self._reset_sim()
    
    def _reset_sim(self):
        """Reset the simulation without reconnecting PyBullet."""
        p.resetSimulation(physicsClientId=self.client_id)
        p.setTimeStep(self.time_step, physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client_id)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.setAdditionalSearchPath(self.ASSETS_PATH, physicsClientId=self.client_id)

        # ground
        try:
            p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        except Exception as e:
            print("[DroneEnv] Warning: cannot load plane.urdf:", e)

        start_pos = [0, 0, 1.0]
        start_ori = p.getQuaternionFromEuler([0, 0, 0])

        self.drone = p.loadURDF(
            "cf2x.urdf",
            basePosition=start_pos,
            baseOrientation=start_ori,
            useFixedBase=False,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.client_id,
        )

        self.step_counter = 0
        start_pos_np = np.array(start_pos, dtype=np.float32)
        self.prev_dist = np.linalg.norm(start_pos_np - self.target)

    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state and return the initial observation."""
        super().reset(seed=seed)
        self._reset_sim()
        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        """
        Apply action and return the new observation, reward, done, and info.
        1. clip action between 0 and 1
        2. convert action to forces
        3. apply forces to drone motors
        4. step simulation
        5. get new observation
        6. compute reward
        7. check termination
        """
        
        action = np.clip(action, 0, 1)

        max_force = 5.0
        forces = action * max_force

        for i in range(4):
            p.applyExternalForce(
                self.drone,
                linkIndex=i,
                forceObj=[0, 0, float(forces[i])],
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
                physicsClientId=self.client_id,
            )

        p.stepSimulation(physicsClientId=self.client_id)
        self.step_counter += 1

        obs = self._get_observation()
        reward = self._compute_reward(obs)
        terminated = self._is_terminated(obs)
        truncated = self.step_counter >= self.max_episode_steps

        if terminated:
            reward -= 10.0 # heavy penalty for crashing

        return obs, float(reward), terminated, truncated, {}

    def _get_observation(self):
        """
        Get the current observation of the drone.
        Observation includes position, orientation (Euler), linear and angular velocities.
        """
        pos, orn = p.getBasePositionAndOrientation(self.dronen, physicsClientId=self.client_id)
        rpy = p.getEulerFromQuaternion(orn)
        lin_vel, ang_vel = p.getBaseVelocity(self.drone, physicsClientId=self.client_id)

        obs = np.array([
            pos[0], pos[1], pos[2],
            rpy[0], rpy[1], rpy[2],
            lin_vel[0], lin_vel[1], lin_vel[2],
            ang_vel[0], ang_vel[1], ang_vel[2]
        ], dtype=np.float32)

        return obs

    def _compute_reward(self, obs):
        """
        Compute the reward based on the drone's position relative to the target.
        """
        pos = obs[0:3]
        dist = np.linalg.norm(pos - self.target)
        
        progress = self.prev_dist - dist

        time_penalty = 0.001 # small penalty to encourage faster stabilization (not stalling)

        reward = progress - time_penalty

        # Bonus for being close to target
        if dist < 0.05:
            reward += 5.0

        self.prev_dist = dist
        return float(reward)

    def _is_terminated(self, obs):
        """
        Termination conditions on z position: 
        - Crash (z < 0.05)
        - Too high (z > 3.0)
        """
        z = obs[2]

        if z < 0.05:
            return True
        
        if z > 3.0:
            return True

        return False

    def render(self):
        """
        Render the environment.
        """
        if self.render_mode == "rgb_array":
            width, height, rgb, _, _ = p.getCameraImage(400,400, physicsClientId=self.client_id)
            return np.array(rgb)
        return None

    def close(self):
        p.disconnect()
