# Recipe planning
from recipe_planner.utils import *

# Navigation planning
import navigation_planner.utils as nav_utils
from navigation_planner.utils import MinPriorityQueue as mpq

# Other core modules
from utils.world import World
from utils.core import *

from collections import defaultdict
import numpy as np
import scipy as sp
import random
from itertools import product
import copy
import time

class QLEARNING:
    """
    Selecting next action 
    """

    def __init__(self):
        """
        """
        self.repr_to_env_dict = dict()
        self.start = None

    def get_next_action(self, env, qtable):
        """Return next action."""

        # Configure planner settings.
        self.start = copy.copy(env)

        # self.start is a copy of the environment.
        cur_state = copy.copy(self.start) 
        #cur_state.display()
        cur_state_encoded = cur_state.encode()
        #print('current state encoded: ', cur_state_encoded)
        stateNr = int(cur_state_encoded, 2)
        # print("state nr: ", stateNr)

        action = np.argmax(qtable[stateNr,:])

        return action

