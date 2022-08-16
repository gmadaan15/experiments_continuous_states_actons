from gym.envs.registration import register
register(id='pendulum_env-v0',entry_point='gym_envs.envs:PendulumEnv',)
register(id='double_integrator_env-v0',entry_point='gym_envs.envs:DoubleIntegratorEnv',)
register(id='pendulum_env-v1',entry_point='gym_envs.envs:PendulumEnvV1',)