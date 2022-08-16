import gym
import torch
from torch.utils.tensorboard import SummaryWriter
#from rl_agent_tiles_discrete_actions import RlAgentTiles
from rl_agent_tiles import RlAgentTiles
from rl_agent_ins import RlAgentIns
from rl_agent_caba import  RlAgentCaba
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 11]
import matplotlib
matplotlib.style.use('seaborn')


import os

def get_most_recent_model(gym_env_name):
    directory_path = "models/{}".format(gym_env_name)

    files = os.listdir(directory_path)
    max_steps_till_now = float("-inf")
    selected_average_reward = 0
    selected_max_episode_reward = 0
    selected_num_episodes = 0
    most_recent_model_file = ""

    for file in files:
        x = file.split("_")
        average_reward, max_episode_reward, num_episodes, steps_till_now, _, _ = file.split("_")

        steps_till_now = int(steps_till_now)
        if steps_till_now > max_steps_till_now:
            max_steps_till_now = steps_till_now
            selected_average_reward = average_reward
            selected_max_episode_reward = max_episode_reward
            selected_num_episodes = num_episodes
            most_recent_model_file = file

    #most_recent_model_file = "most_recent.pth"
    return float(selected_average_reward), float(selected_max_episode_reward), int(selected_num_episodes), int(max_steps_till_now),  "{}/{}".format( directory_path, most_recent_model_file)

def save_model (gym_env_name,agent, steps_till_now, num_episodes, average_reward, max_episode_reward):
    directory_path = "models/{}".format(gym_env_name)

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    agent.save("models/{}/{}_{}_{}_{}_".format(gym_env_name, average_reward, max_episode_reward, num_episodes, steps_till_now))


def load_model(gym_env_name,agent, steps_till_now, num_episodes, average_reward, max_episode_reward):
    agent.load("models/{}/{}_{}_{}_{}_".format(gym_env_name, average_reward, max_episode_reward, num_episodes, steps_till_now))

# taken from the paper
def experiment(gym_env_name , hyperparameters ):


    env = gym.make(gym_env_name)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    agent = RlAgentIns()#Rl_Agent_Tiles()
    agent.agent_init(hyperparameters)


    last_obs = env.reset()
    action = agent.agent_start(last_obs)

    num_rep = 2
    num_episodes = 200
    num_steps = 200

    average_reward_list = [0]*num_episodes
    variance_reward_list = [0]*num_episodes
    variance_steps_list = [0]*num_episodes
    average_steps_list = [0]*num_episodes


    for i in range(num_rep):
        agent.agent_init(hyperparameters)
        for j in range(num_episodes):
            episode_reward = 0
            last_obs = env.reset()
            action = agent.agent_start(last_obs)
            for k in range(num_steps):
                env.render()

                obs, reward, done, info = env.step(action)
                episode_reward += reward
                if done:
                    episode_reward = float(episode_reward)
                    agent.agent_end(reward)
                    break
                action = agent.agent_step(reward, obs)

            average_reward_list[j]+= episode_reward
            variance_reward_list[j] += episode_reward**2
            average_steps_list[j] += float(k)
            variance_steps_list[j] += float(k**2)

    average_reward_list = np.array(average_reward_list)
    average_steps_list = np.array(average_steps_list)
    variance_steps_list = np.array(variance_steps_list)
    variance_reward_list = np.array(variance_reward_list)

    num_rep = float(num_rep)

    average_reward_list /= num_rep
    average_steps_list /= num_rep

    variance_reward_list = variance_reward_list - num_rep*(average_reward_list**2)
    variance_reward_list = variance_reward_list/((num_rep-1)*num_rep)

    variance_steps_list = variance_steps_list - num_rep * (average_steps_list ** 2)
    variance_steps_list = variance_steps_list / ((num_rep - 1) * num_rep)


    fig, axes = plt.subplots(1,2)

    axes[0].scatter(average_reward_list, range(1,num_episodes+1))
    #axes[0].fill_between(average_reward_list, average_reward_list + variance_reward_list**(0.5), average_reward_list - variance_reward_list**(0.5),
                            #color='red', alpha=0.15)

    axes[1].scatter(average_steps_list, range(1,num_episodes+1))
    #axes[1].fill_between(average_steps_list, average_steps_list + variance_steps_list ** (0.5),
                            #average_steps_list - variance_steps_list ** (0.5),
                            #color='red', alpha=0.15)

    plt.show()

def train(gym_env_name = "Pendulum-v0", continue_training = True, hyperparameters = {} ):

    env = gym.make(gym_env_name)
    obs_size = env.observation_space.shape[0]
    #n_actions = env.action_space.shape[0]

    agent = RlAgentCaba()#RlAgentTiles()#RlAgentCaba()#RlAgentTiles()#RlAgentIns()
    agent.agent_init(hyperparameters)
    steps_till_now = 0
    max_episode_reward = float("-inf")
    num_episodes = 0
    average_reward = 0
    if continue_training == True:
        average_reward, max_episode_reward, num_episodes, steps_till_now, file_path =  get_most_recent_model(gym_env_name)
        load_model(gym_env_name,agent, steps_till_now, num_episodes, average_reward, max_episode_reward)


    summary_writer = SummaryWriter('runs/{}'.format(gym_env_name))
    episode_reward = 0

    last_obs = env.reset()
    action = agent.agent_start(last_obs)

    while(max_episode_reward < 3000):
        env.render()

        obs, reward, done, info = env.step(action)
        episode_reward += reward

        steps_till_now +=1

        if done or steps_till_now >= 200:
            episode_reward = float(episode_reward)
            num_episodes+=1
            average_reward = average_reward + (1/num_episodes)*( episode_reward - average_reward )
            #summary_writer.add_scalar(gym_env_name + "/average_reward vs num_steps", average_reward, steps_till_now)
            summary_writer.add_scalar(gym_env_name + "/average_reward vs num_episodes", average_reward, num_episodes)

            #episode_reward = steps_till_now
            if episode_reward > max_episode_reward:
                print(episode_reward)
                max_episode_reward = episode_reward

            #summary_writer.add_scalar(gym_env_name + "/maximum_episode_reward vs num_steps", max_episode_reward,
            #                          steps_till_now)
            summary_writer.add_scalar(gym_env_name + "/maximum_episode_reward vs num_episodes", max_episode_reward,
                                      num_episodes)

            save_model(gym_env_name, agent, steps_till_now, num_episodes, average_reward, max_episode_reward)
            episode_reward = 0
            steps_till_now = 0
            agent.agent_end(reward)
            last_obs = env.reset()
            action = agent.agent_start(last_obs)
            continue

        action = agent.agent_step(reward, obs)

    summary_writer.close()


def test (gym_env_name = "Pendulum-v0", hyperparameters={}):
    # Make the environment and model, and train
    env = gym.make(gym_env_name)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    actor_critic = FeedForwardNN(obs_size, n_actions)
    steps_per_batch = hyperparameters["steps_per_batch"]

    _,model_path = get_most_recent_model(gym_env_name)
    actor_critic.load_state_dict(torch.load(model_path))

    max_episode_reward = float("-inf")
    num_episodes = 0
    average_reward = 0
    episode_reward = 0

    while(True):
        last_obs = env.reset()
        for step in range(steps_per_batch):

            env.render()

            value, action, log_probs = actor_critic.act(torch.tensor(last_obs).float())

            obs, reward, done, info = env.step(np.array(action).copy())
            episode_reward += reward

            if done:
                episode_reward = float(episode_reward)
                num_episodes += 1
                average_reward = average_reward + (1 / num_episodes) * (episode_reward - average_reward)
                obs = env.reset()
                if episode_reward > max_episode_reward:
                    print(episode_reward)
                    max_episode_reward = episode_reward
                episode_reward = 0

            last_obs = obs
        print(average_reward)
        average_reward = 0
        num_episodes = 0



if __name__ == '__main__':

    env = 'gym_envs:double_integrator_env-v0'

    if env == 'gym_envs:pendulum_env-v0':
        hyperparameters = { 'gamma': 0.99, 'lamda': 0.7, 'alpha': 0.4, "epsilon": 0.05, 'threshold': 0.065, 'fraction':0.6,
                           "state_dist_kernel":0.065, "action_dist_kernel": 0.2, 'max_num_cases':2000,
                            "state_info":{"dims":2, 'input_ranges': [(-np.pi, np.pi), (-2.0*np.pi, +2.0*np.pi)] },
                            "action_info":{ "dims":1, "input_ranges":[(-3.0, +3.0)], "num_action_divisions": 25, "state_num_action_divisions":6 }}

    if env == 'gym_envs:double_integrator_env-v0':
        hyperparameters = {'gamma': 0.99, 'lamda': 0.7, 'alpha': 0.1, "epsilon": 0.00, 'threshold': 0.065, 'fraction':0.6,
                           "state_dist_kernel":0.065, "action_dist_kernel": 0.2, "kernel":0.065,
                           'max_num_cases': 2000,"state_info": {"dims": 2, 'input_ranges': [(-1.0, 1.0), (-1.0, 1.0)]},
                           "action_info": {"dims": 1, "input_ranges": [(-1.0, +1.0)], "num_action_divisions": 25, "state_num_action_divisions":6}}
    if env == "Pendulum-v1":
        hyperparameters = {'gamma': 0.99, 'lamda': 0.8, 'alpha': 0.1, "epsilon": 0.08, 'threshold': 0.065, 'fraction':0.6,
                           "state_dist_kernel":0.065, "action_dist_kernel": 0.2,
                           'max_num_cases': 5000,"state_info": {"dims": 3, 'input_ranges': [(-1.0, 1.0), (-1.0, 1.0), (-8.0, 8.0)]},
                           "action_info": {"dims": 1, "input_ranges": [(-2.0, +2.0)], "num_action_divisions": 50, "state_num_action_divisions":6 }}

    if env == 'gym_envs:pendulum_env-v1':
        hyperparameters = { 'gamma': 0.99, 'lamda': 0.7, 'alpha': 0.4, "epsilon": 0.05, 'threshold': 0.065, 'fraction':0.6,
                           "state_dist_kernel":0.065, "action_dist_kernel": 0.2, 'max_num_cases':2000,
                            "state_info":{"dims":2, 'input_ranges': [(-np.pi, np.pi), (-8.0,8.0)] },
                            "action_info":{ "dims":1, "input_ranges":[(-2.0, +2.0)], "num_action_divisions": 25, "state_num_action_divisions":8 }}
    # use discrete actions tiles for this.
    if env == 'MountainCar-v0':
        hyperparameters = {'gamma': 0.99, 'lamda': 0.7, 'alpha': 0.4, "epsilon": 0.05, 'threshold': 0.065,
                           "state_info": {"dims": 5, 'input_ranges': [(-0.3, 14.4)]*361+ [(-0.07, 0.07)]},
                           "action_info": {"num_actions": 24}}

    train(gym_env_name= env, continue_training=False, hyperparameters=hyperparameters)
    experiment(gym_env_name= env, hyperparameters=hyperparameters)


