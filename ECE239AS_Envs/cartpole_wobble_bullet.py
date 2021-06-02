
import numpy as np

import gym
from gym import spaces
from gym.envs.registration import registry

from pybullet_envs.bullet import CartPoleContinuousBulletEnv

id = 'CartPoleWobbleContinuousEnv-v0'
class CartPoleWobbleContinuousEnv(CartPoleContinuousBulletEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

        # Keep track of target location
        self.target_pos = 0.5
        self.target_threshold = 0.1
        self.target_reward = 10

        # Override x_threshold of CartPoleContinuousBulletEnv
        self.x_threshold = 2.4  #Original: 0.4
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max, self.theta_threshold_radians * 2,
            np.finfo(np.float32).max
        ])

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def step(self, action):
        raw_state, reward, done, info = super().step(action)

        theta, theta_dot, x, x_dot = raw_state
        print(theta)
        # Check for target approach
        x_good = np.abs(x - self.target_pos) < self.target_threshold
        t_good = np.abs(theta) < 1 * (3.1415926 / 180.0) # Degrees to Radians
        if done:
            reward = -self.target_reward
        elif x_good and t_good:
            reward = self.target_reward
            done = True
        else:
            reward = 0

        # Package target into state
        state = raw_state

        return state, reward, done, info

    def reset(self):
        state = super().reset()
        # self.target =
        return state

if not id in registry.env_specs:
    gym.envs.registration.register(id,
        entry_point='ECE239AS_Envs:CartPoleWobbleContinuousEnv',
        max_episode_steps=10000)
