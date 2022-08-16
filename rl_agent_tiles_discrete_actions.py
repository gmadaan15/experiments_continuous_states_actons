import numpy as np
from tile_coder import Tile_Coder
from PyFixedReps import TileCoder
from common_utils import *


# rl agent with continuous state and action space using tiles as a function
# approximator, basically we are using tiles to accomodate the big continuous
# space and rest q function will work as linear function of this tile
# representation
class RlAgentTiles:

    # the common things that will be required
    def __init__(self):
        self.last_action = None
        self.last_state = None
        self.epsilon = None
        self.gamma = None
        self.w = None
        self.e = None
        self.alpha = None
        self.previous_tiles = None
        self.lamda = None
        self.num_actions = None

    # agent initialisation, agent_info will contain all info
    # dict
    def agent_init(self, agent_info):
        self.epsilon = agent_info.get("epsilon", 0.92)
        self.gamma = agent_info.get("gamma", 0.97)
        self.alpha = agent_info.get("alpha", 0.1)
        self.lamda = agent_info.get("lamda", 0.7)
        initial_weights = agent_info.get("initial_weights", 0)
        state_info = agent_info["state_info"]
        action_info = agent_info["action_info"]

        self.num_actions = action_info["num_actions"]


        self.__create_tiles_encoder__(state_info, initial_weights)

    def __create_tiles_encoder__(self, state_info, initial_weights):

        dims = state_info["dims"]
        tiles = 4 * dims
        tilings = int(2 ** np.ceil(np.log2(tiles)))
        input_ranges = state_info["input_ranges"]
        params = {
            'dims': dims,
            'tiles': tiles,
            'tilings': tilings,
            'input_ranges': input_ranges
        }

        self.tc1 = TileCoder(params)
        self.tc = lambda x: self.tc1.get_indices(x)
        self.tiles_len = tilings
        self.w = np.ones((self.num_actions, self.tc1.features())) * initial_weights
        print(self.w.nbytes)
        t = 0


    def select_action(self, state_tiles):
        """
        Selects an action using epsilon greedy
        Args:
        state - float values list
        Returns:
        (chosen_action, action_value) - (int, float), tuple of the chosen action
                                        and it's value
        """

        action_values = []
        chosen_action = None

        for i in range(self.num_actions):
            action_values.append(np.sum(self.w[i][state_tiles]))

        if np.random.random() < self.epsilon:
            chosen_action = np.random.choice(range(self.num_actions))
        else:
            chosen_action = argmax(action_values)

        return chosen_action, action_values[chosen_action]

    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state observation from the
                environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        state_tiles = self.tc(state)
        current_action, _ = self.select_action(state_tiles)



        self.last_action = current_action
        self.previous_tiles = np.copy(state_tiles)
        return self.last_action

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """
        active_tiles = self.tc(state)
        current_action, action_value = self.select_action(active_tiles)

        previous_value = np.sum(self.w[self.last_action][self.previous_tiles])

        self.w[self.last_action][self.previous_tiles] += self.alpha/self.tiles_len * (
                    reward + self.gamma * action_value - previous_value)


        self.last_action = current_action
        self.previous_tiles = np.copy(active_tiles)
        return self.last_action

    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        # Update self.w at self.previous_tiles and self.previous action
        # using the reward, self.gamma, self.w,
        # self.alpha, and the Sarsa update from the textbook
        # Hint - there is no action_value used here because this is the end
        # of the episode.

        previous_value = np.sum(self.w[self.last_action][self.previous_tiles])
        self.w[self.last_action][self.previous_tiles] += self.alpha/self.tiles_len * (reward - previous_value)


    def save(self, file_name=""):
        np.save(file_name + "_w", self.w)


    def load(self, file_name=""):
        self.w = np.load(file_name + "_w" + ".npy")



