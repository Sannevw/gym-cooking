# Recipe planning
from recipe_planner.stripsworld import STRIPSWorld
import recipe_planner.utils as recipe_utils
from recipe_planner.utils import *

# Navigation planner
from navigation_planner.planners.qlearning import QLEARNING
import navigation_planner.utils as nav_utils

# Other core modules
from utils.core import Counter, Cutboard


import numpy as np
import copy
import random
from termcolor import colored as color
from collections import namedtuple

AgentRepr = namedtuple("AgentRepr", "name location holding")
SubtaskAllocation = namedtuple("SubtaskAllocation", "subtask subtask_agent_names")

# Colors for agents.
COLORS = ['white', 'blue', 'magenta', 'yellow', 'green']
ACTION_TO_NAME = {(0, 1): 0, (0, -1): 1, (-1, 0): 2, (1, 0): 3} # (0, 0): 4}

class RealAgent:
    """Real Agent object that performs task inference and plans."""

    def __init__(self, arglist, name, id_color, recipes):
        self.arglist = arglist
        self.name = name
        self.color = id_color
        self.recipes = recipes

        self.planner = QLEARNING()


    def __str__(self):
        return color(self.name[-1], self.color)

    def __copy__(self):
        a = Agent(arglist=self.arglist,
                name=self.name,
                id_color=self.color,
                recipes=self.recipes)
        a.__dict__ = self.__dict__.copy()
        if self.holding is not None:
            a.holding = copy.copy(self.holding)
        return a

    def get_holding(self):
        if self.holding is None:
            return 'None'
        return self.holding.full_name

    def init_action(self, obs):
        sim_agent = list(filter(lambda x: x.name == self.name, obs.sim_agents))[0]
        self.location = sim_agent.location
        self.holding = sim_agent.holding
        self.action = sim_agent.action
        
    def select_action(self, obs, qtable, epsilon):
        """Return best next action for this agent given observations."""
        rand_nr = random.uniform(0,1)
        if rand_nr > epsilon:
            self.plan(copy.copy(obs),copy.copy(qtable))
        else:
            self.action = ACTION_TO_NAME[random.choice(obs.world.NAV_ACTIONS)]        
        return self.action

    def get_action_location(self):
        """Return location if agent takes its action---relevant for navigation planner."""
        return tuple(np.asarray(self.location) + np.asarray(self.action))

    def plan(self, env, qtable):
        # get argmax, biggest q value for this state.
        self.action = self.planner.get_next_action(
                env=env, qtable=qtable)

class SimAgent:
    """Simulation agent used in the environment object."""

    def __init__(self, name, id_color, location):
        self.name = name
        self.color = id_color
        self.location = location
        self.holding = None
        self.action = (0, 0)
        self.has_delivered = False

    def __str__(self): 
        return color(self.name[-1], self.color)

    def __copy__(self):
        a = SimAgent(name=self.name, id_color=self.color,
                location=self.location)
        a.__dict__ = self.__dict__.copy()
        if self.holding is not None:
            a.holding = copy.copy(self.holding)
        return a

    def get_repr(self):
        return AgentRepr(name=self.name, location=self.location, holding=self.get_holding())

    def get_holding(self):
        if self.holding is None:
            return 'None'
        return self.holding.full_name

    def print_status(self):
        if self.color == 'robot':
            print("{} currently at {}, action {}, holding {}".format(
                color('robot', 'grey'),
                self.location,
                self.action,
                self.get_holding()))
        else:   
            print("{} currently at {}, action {}, holding {}".format(
                    color(self.name, self.color),
                    self.location,
                    self.action,
                    self.get_holding()))

    def acquire(self, obj):
        if self.holding is None:
            self.holding = obj
            self.holding.is_held = True
            self.holding.location = self.location
        else:
            self.holding.merge(obj) # Obj(1) + Obj(2) => Obj(1+2)

    def release(self):
        self.holding.is_held = False
        self.holding = None

    def move_to(self, new_location):
        self.location = new_location
        if self.holding is not None:
            self.holding.location = new_location
