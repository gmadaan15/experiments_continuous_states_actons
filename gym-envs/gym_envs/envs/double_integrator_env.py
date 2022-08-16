import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import math


class DoubleIntegratorEnv(gym.Env):

    def __init__(self, mode = "start"):
        self.max_vel = 1.0
        self.max_acc = 1.0
        self.max_pos = 1.0
        self.dt = 0.05
        self.num_dts_per_step = 4

        self.pos_0 = 1.0
        self.vel_0 = 0.0

        self.pos_f = 0.0
        self.vel_f = 0.0

        self.time = 0.0

        self.min_reward = -50.0

        self.modes_dict = {"start":0, "goal": 1, "random":2}
        self.mode = self.modes_dict[mode]

        high = np.array([self.max_pos, self.max_vel], dtype=np.float32)
        self.action_space = spaces.Box(
            low = -self.max_acc,
            high = self.max_acc, shape=(1,),
            dtype = np.float32
        )
        self.observation_space = spaces.Box(
            low = -high,
            high = high,
            dtype = np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        pos, vel = self.state  # th := theta

        dt = self.dt
        r = 0.0

        # u = np.clip(u, -self.max_torque, self.max_torque)[0]
        u = u[0]
        for i in range(self.num_dts_per_step):
            delta_pos = self.pos_f - pos
            delta_vel = self.vel_f - vel


            r += -(delta_pos * delta_pos + u * u)

            vel += (u * dt)
            pos += (vel * dt)

            self.time += dt

        self.last_u = u
        self.state = np.array([pos, vel])

        if  (pos < -self.max_pos) or (pos > self.max_pos) or (vel < -self.max_vel) or (vel >  self.max_vel):
            reward = self.min_reward
            return self._get_obs(), reward, True, {}

        reward = r * dt
        return self._get_obs(), reward, False, {}



    def reset(self):
        if self.mode == self.modes_dict["start"]:
            self.state = np.array([self.pos_0, self.vel_0])

        elif self.mode == self.modes_dict["goal"]:
            self.state = np.array([self.pos_f, self.vel_f])

        elif self.mode == self.modes_dict["random"]:
            high = np.array([self.max_pos, self.max_vel])
            self.state = self.np_random.uniform(low=-high, high=high)

        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        pos, vel = self.state
        return self.state
    def render(self, mode='human'):
        pass

