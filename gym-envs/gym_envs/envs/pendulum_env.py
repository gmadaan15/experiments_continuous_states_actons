import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import math


class PendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, g=9.8, mode = "start"):
        self.max_speed = 2.0*np.pi
        self.max_torque = 3.0
        self.max_theta = np.pi
        self.dt = .05
        self.num_dts_per_step = 4
        self.g = g
        self.m = 1.0/3.0
        self.l = 3.0/2.0
        self.viewer = None


        self.theta_0 = -np.pi
        self.thdot_0 = 0

        self.theta_f = 0.0
        self.thdot_f = 0.0

        self.time = 0.0

        self.min_reward = -2500.0

        self.modes_dict = {"start":0, "goal": 1, "random":2}
        self.mode = self.modes_dict[mode]

        high = np.array([self.max_theta, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low = -self.max_torque,
            high = self.max_torque, shape=(1,),
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
        theta_out, thdot_out = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        r = 0.0

        #u = np.clip(u, -self.max_torque, self.max_torque)[0]
        u = u[0]

        for i in range(self.num_dts_per_step):
            delta_theta = self.theta_f - angle_normalize(theta_out, self.theta_f)
            delta_thdot = self.thdot_f - thdot_out

            r += -(delta_theta * delta_theta + delta_thdot * delta_thdot + u*u);

            tmp = angle_normalize(theta_out, 0);
            sin_tmp = np.sin(tmp)
            thdot_out += 3 * ( u + m * g * l * sin_tmp ) / (4 * m * l * l) * dt;

            theta_out += thdot_out * dt;

            self.time += dt

        self.last_u = u
        self.state = np.array([theta_out, thdot_out])

        if thdot_out < -self.max_speed or thdot_out > self.max_speed:
            reward = self.min_reward
            return self._get_obs(), reward, True, {}

        reward = r*dt
        return self._get_obs(), reward, False, {}

    def reset(self):
        if self.mode == self.modes_dict["start"]:
            self.state = np.array([self.theta_0, self.thdot_0])
        elif self.mode == self.modes_dict["goal"]:
            self.state = np.array([self.theta_f, self.thdot_f])

        elif self.mode == self.modes_dict["random"]:
            high = np.array([self.max_theta, self.max_speed])
            self.state = self.np_random.uniform(low=-high, high=high)

        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return self.state

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x, center):
    min = center - np.pi
    max = center + np.pi
    tmp = math.fmod(x - min, max - min)
    if tmp >= 0.0:
        return min + tmp;
    else:
        return max + tmp;
