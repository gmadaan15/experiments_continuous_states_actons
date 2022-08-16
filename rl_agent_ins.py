import numpy as np
from tile_coder import Tile_Coder
from PyFixedReps import TileCoder
from common_utils import *

class Case:
    def __init__(self, state, action, q_value, eligibility):
        self.state = state
        self.action = action
        self.q_value = q_value
        self.eligibility = eligibility

class Neighbour:
    def __init__(self, idx, dist, weight):
        self.idx = idx
        self.dist = dist
        self.weight = weight


# rl agent with continuous state and action space using instance based memory as a function
# approximator,
class RlAgentIns:

    # the common things that will be required
    def __init__(self):
        self.last_action = None
        self.last_state = None
        self.epsilon = None
        self.gamma = None
        self.e = None
        self.alpha = None
        self.lamda = None

        self.threshold = None
        self.kernel = None
        self.division_factor = None

        self.max_num_cases = None
        self.cases = None


    # agent initialisation, agent_info will contain all info
    # dict
    def agent_init(self, agent_info):
        self.epsilon = agent_info.get("epsilon", 0.05)
        self.gamma = agent_info.get("gamma", 0.97)
        self.alpha = agent_info.get("alpha", 0.1)
        self.lamda = agent_info.get("lamda", 0.7)
        self.threshold = agent_info.get('threshold', 0.065)
        self.kernel = agent_info.get("kernel", 0.065)

        self.division_factor = 0
        action_info = agent_info["action_info"]
        action_min = []
        action_max = []
        for i in range(action_info['dims']):
            min_a, max_a = action_info['input_ranges'][i]
            action_min.append(min_a)
            action_max.append(max_a)
            self.division_factor += (max_a - min_a)**2

        action_max = np.array(action_max)
        action_min = np.array(action_min)

        # helps in one step search
        num_action_divisions = action_info["num_action_divisions"]
        delta_action = (action_max - action_min)/num_action_divisions
        self.search_actions = []
        for i in range(num_action_divisions):
            action = action_min + delta_action*0.5 + i*delta_action
            self.search_actions.append(action)

        state_info = agent_info["state_info"]

        for i in range(state_info['dims']):
            min_s, max_s = state_info['input_ranges'][i]
            self.division_factor += (max_s - min_s)**2

        self.division_factor = self.division_factor**(0.5)

        self.max_num_cases = agent_info["max_num_cases"]
        self.cases = np.array([])

    def __match_case__(self, point1, point2):
        dist = np.linalg.norm(point1 - point2)
        dist /= self.division_factor
        return dist

    def __select_neighbours__(self, state, action):
        state_action = np.concatenate((state, action), axis=None)

        neighbours = []
        total_weight = 0
        num_cases = len(self.cases)
        for j in range(num_cases):
            case = self.cases[j]
            state_action_j = np.concatenate((case.state, case.action), axis=None)
            dist = self.__match_case__(state_action, state_action_j)

            if dist < self.threshold:
                weight = np.exp(- (dist*dist)/(self.kernel*self.kernel))
                neighbour = Neighbour(j, dist, weight)
                neighbours.append(neighbour)
                total_weight += weight

        for neighbour in neighbours:
            neighbour.weight /= total_weight

        return neighbours

    def ___compute_q_value__(self, neighbours):
        q_value = 0
        for neighbour in neighbours:
            q_value += self.cases[neighbour.idx].q_value * neighbour.weight
        return q_value

    def __add_case__(self, state, action, num_neighbours):
        num_cases = len(self.cases)

        if num_cases == self.max_num_cases:
            return

        case = Case(state, action, 0.0, 0.0)
        if num_cases == 0:
            self.cases = np.array([case])
            return

        # added only if num_neighbours is zero
        if num_neighbours == 0:
            self.cases = np.append(self.cases, case)

    def __clear_eligibilities__(self):
        num_cases = len(self.cases)
        for i in range(num_cases):
            self.cases[i].eligibility = 0

    def __update_eligibilities__(self):
        num_cases = len(self.cases)
        for i in range(num_cases):
            self.cases[i].eligibility *= (self.gamma * self.lamda)

    def __replace_eligibilities__(self, neighbours):
        for neighbour in neighbours:
            self.cases[neighbour.idx].eligibility = neighbour.weight

    def __update_value_function__(self, value):
        num_cases = len(self.cases)
        for i in range(num_cases):
            self.cases[i].q_value += (value * self.cases[i].eligibility)

    def select_action(self, state):
        """
        Selects an action using epsilon greedy
        Args:
        state - float values list
        Returns:
        (chosen_action, action_value) - (int, float), tuple of the chosen action
                                        and it's value
        """
        chosen_action = None

        # no cases, just select a random action and add it as a case
        num_cases = len(self.cases)
        if num_cases == 0:
            chosen_action = np.random.choice(range(len(self.search_actions)))
            return self.search_actions[chosen_action], 0, 0

        # else select the maximum value action using epsilon greedy

        action_values = []
        num_neighbours = []
        for i in range(len(self.search_actions)):
            neighbours = self.__select_neighbours__(state, self.search_actions[i])
            q_value = self.___compute_q_value__(neighbours)
            action_values.append(q_value)
            num_neighbours.append(len(neighbours))

        # epsilon greedy selection
        if np.random.random() < self.epsilon:
            chosen_action = np.random.choice(range(len(self.search_actions)))
        else:
            chosen_action = argmax(action_values)

        # since this has been selected, it needs to checked if this should be added to the
        # list of cases or not
        #self.__add_case__(state, self.search_actions[chosen_action], num_neighbours[chosen_action])
        return (self.search_actions[chosen_action], action_values[chosen_action], num_neighbours[chosen_action])

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

        (current_action, value , num_neighbours)= self.select_action(state)
        self.last_action = current_action
        self.last_state = state

        self.__add_case__(state, current_action, num_neighbours)
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
        neighbours = self.__select_neighbours__(self.last_state, self.last_action)
        previous_value = self.___compute_q_value__(neighbours)

        self.__replace_eligibilities__(neighbours)

        current_action, current_value, num_neighbours = self.select_action(state)

        delta_q = self.alpha*(reward + self.gamma*current_value - previous_value)

        self.__update_value_function__(delta_q)

        self.last_action = current_action
        self.last_state = state
        self.__add_case__(state, current_action, num_neighbours)
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
        neighbours = self.__select_neighbours__(self.last_state, self.last_action)
        previous_value = self.___compute_q_value__(neighbours)

        self.__replace_eligibilities__(neighbours)

        current_value =0

        delta_q = self.alpha * (reward + self.gamma * current_value - previous_value)

        self.__update_value_function__(delta_q)

    def save(self, file_name = ""):
        np.save(file_name + "_w", self.cases)


    def load(self, file_name = ""):
        self.cases = np.load(file_name + "_w" + ".npy")



