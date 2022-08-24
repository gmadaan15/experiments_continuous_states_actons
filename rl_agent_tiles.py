import numpy as np
#from tile_coder import Tile_Coder
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

    # agent initialisation, agent_info will contain all info
    # dict
    def agent_init(self, agent_info):
        self.epsilon = agent_info.get("epsilon", 0.95)
        self.gamma = agent_info.get("gamma", 0.97)
        self.alpha = agent_info.get("alpha", 0.1)
        self.lamda = agent_info.get("lamda", 0.7)
        initial_weights = agent_info.get("initial_weights", 0)
        state_info = agent_info["state_info"]
        action_info = agent_info["action_info"]

        # helps in one step search
        action_min = []
        action_max = []
        for i in range(action_info['dims']):
            min_a, max_a = action_info['input_ranges'][i]
            action_min.append(min_a)
            action_max.append(max_a)

        action_max = np.array(action_max)
        action_min = np.array(action_min)

        # helps in one step search
        num_action_divisions = action_info["num_action_divisions"]
        delta_action = (action_max - action_min) / num_action_divisions
        self.search_actions = []
        for i in range(num_action_divisions):
            action = action_min + delta_action * 0.5 + i * delta_action
            self.search_actions.append(action)


        self.__create_tiles_encoder__(state_info, action_info, initial_weights )

    def __create_tiles_encoder__(self, state_info, action_info, initial_weights):

        dims = state_info["dims"] + action_info["dims"]
        tiles = 4*dims
        tilings = int(2**np.ceil(np.log2(tiles)))
        input_ranges = state_info["input_ranges"] + action_info["input_ranges"]
        params = {
                'dims': dims,
                'tiles': tiles,
                'tilings': tilings,
                'input_ranges': input_ranges
            }

        '''
        params = {
                'dims': 3,
                'tiles': 12,
                'tilings': 36,
                'input_ranges': [(-np.pi, np.pi), (-2.0*np.pi, +2.00*np.pi), (-3.0, +3.00)]
            }
        
        self.tc1 = Tile_Coder( params )

        params = {
                'dims': 2,
                'tiles': 12,
                'tilings': 12,
                'input_ranges': [(-np.pi, np.pi), (-8.0, +8.0)]
            }

        self.tc2 = Tile_Coder(params)

        params = {
            'dims': 1,
            'tiles': 12,
            'tilings': 12,
            'input_ranges': [(-2.0, 2.0)]
        }

        self.tc3 = Tile_Coder(params)

        self.tc = lambda state, action : self.tc1.get_indices(state + [action]) + self.tc2.get_indices(state) + self.tc3.get_indices(action)

        self.w = np.ones(self.tc1.features() + self.tc2.features() + self.tc3.features()) * initial_weights

        self.e =  np.ones(self.tc1.features() + self.tc2.features() + self.tc3.features()) * 0

        self.tiles_len = 36
        '''

        self.tc1 = TileCoder( params )
        self.tc = lambda x: self.tc1.get_indices(x)

        self.w = np.ones(self.tc1.features()) * initial_weights

        self.e = np.ones(self.tc1.features()) * 0

        self.tiles_len = tilings

    def __clear_eligibilities__(self):
        self.e.fill(0)

    def __update_eligibilities__(self):
        self.e = self.e * self.lamda * self.gamma

    def __replace_eligibilities__(self):
        self.e[self.previous_tiles] = 1/self.tiles_len

    def __update_value_function__(self, value):
        #self.w[self.previous_tiles] += (self.alpha / self.tiles_len) * (reward + self.gamma * action_value - previous_value)
        self.w[self.previous_tiles] += self.e[self.previous_tiles]*value


    def select_action(self, state):
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

        for i in range(len(self.search_actions)):
            state_action = np.concatenate((state, self.search_actions[i]), axis=None)
            tiles = self.tc(state_action)
            action_values.append(np.sum(self.w[tiles]))

        if np.random.random() < self.epsilon:
            chosen_action = np.random.choice(range(len(self.search_actions)))
        else:
            chosen_action = argmax(action_values)

        return self.search_actions[chosen_action], action_values[chosen_action]

    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state observation from the
                environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        self.__clear_eligibilities__()
        current_action, _ = self.select_action(state)

        active_tiles = self.tc(np.concatenate((state, current_action), axis=None))

        self.last_action = current_action
        self.previous_tiles = np.copy(active_tiles)
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
        self.__update_eligibilities__()

        previous_value = np.sum(self.w[self.previous_tiles])

        self.__replace_eligibilities__()

        current_action, action_value = self.select_action(state)
        active_tiles = self.tc(np.concatenate((state, current_action), axis=None))

        delta_q = self.alpha*(reward + self.gamma * action_value - previous_value)

        self.__update_value_function__(delta_q)

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

        self.__update_eligibilities__()

        previous_value = np.sum(self.w[self.previous_tiles])

        self.__replace_eligibilities__()

        action_value = 0

        delta_q = self.alpha * (reward + self.gamma * action_value - previous_value)

        self.__update_value_function__(delta_q)


    def save(self, file_name = ""):
        np.save(file_name + "_w", self.w)
        np.save(file_name + "_e", self.e)

    def load(self, file_name = ""):
        self.w = np.load(file_name + "_w" + ".npy")
        self.e = np.load(file_name + "_e" + ".npy")



