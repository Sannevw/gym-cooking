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
from functools import lru_cache
from enum import Enum

class PlannerLevel(Enum):
    LEVEL1 = 1
    LEVEL0 = 0

def argmin(vector):
    e_x = np.array(vector) == min(vector)

    return np.where(np.random.multinomial(1, e_x / e_x.sum()))[0][0]

def argmax(vector):
    e_x = np.array(vector) == max(vector)
    return np.where(np.random.multinomial(1, e_x / e_x.sum()))[0][0]

class QLEARNING:
    """
    Selecting next action 
    """

    def __init__(self):
        """
        """

        self.v_l = {}
        self.v_u = {}
        self.repr_to_env_dict = dict()
        self.start = None
     
    def _configure_planner_level(self, env, subtask_agent_names, other_agent_planners):
        """Configure the planner s.t. it best responds to other agents as needed.

        If other_agent_planners is an emtpy dict, then this planner should
        be a level-0 planner and remove all irrelevant agents in env.

        Otherwise, it should keep all agents and maintain their planners
        which have already been configured to the subtasks we believe them to
        have."""
        
        self.planner_level = PlannerLevel.LEVEL0
        self.other_agent_planners = {}
        # Replace other agents with counters (frozen agents during planning).
        rm_agents = []
        for agent in env.sim_agents:
            if agent.name not in subtask_agent_names:
                rm_agents.append(agent)
        for agent in rm_agents:
            env.sim_agents.remove(agent)
            if agent.holding is not None:
                self.removed_object = agent.holding
                env.world.remove(agent.holding)

            # Remove Floor and replace with Counter. This is needed when
            # checking whether object @ location is collidable.
            env.world.remove(Floor(agent.location))
            env.world.insert(AgentCounter(agent.location))

    def _configure_subtask_information(self, subtask, subtask_agent_names):
        """Tracking information about subtask allocation."""
        # Subtask allocation
        self.subtask = subtask
        self.subtask_agent_names = subtask_agent_names

        # Relevant objects for subtask allocation.
        self.start_obj, self.goal_obj = nav_utils.get_subtask_obj(subtask)
        self.subtask_action_obj = nav_utils.get_subtask_action_obj(subtask)

    def _define_goal_state(self, env, subtask):
        """Defining a goal state (termination condition on state) for subtask."""

        if subtask is None:
            self.is_goal_state = lambda h: True

        # Termination condition is when desired object is at a Deliver location.
        elif isinstance(subtask, Deliver):
            self.cur_obj_count = len(
                    list(filter(lambda o: o in set(env.world.get_all_object_locs(
                            self.subtask_action_obj)),
                    env.world.get_object_locs(self.goal_obj, is_held=False))))
            self.has_more_obj = lambda x: int(x) > self.cur_obj_count
            self.is_goal_state = lambda h: self.has_more_obj(
                    len(list(filter(lambda o: o in set(env.world.get_all_object_locs(self.subtask_action_obj)),
                    self.repr_to_env_dict[h].world.get_object_locs(self.goal_obj, is_held=False)))))

            if self.removed_object is not None and self.removed_object == self.goal_obj:
                self.is_subtask_complete = lambda w: self.has_more_obj(
                        len(list(filter(lambda o: o in set(env.world.get_all_object_locs(self.subtask_action_obj)),
                        w.get_object_locs(self.goal_obj, is_held=False)))) + 1)
            else:
                self.is_subtask_complete = lambda w: self.has_more_obj(
                        len(list(filter(lambda o: o in set(env.world.get_all_object_locs(self.subtask_action_obj)),
                        w.get_object_locs(obj=self.goal_obj, is_held=False)))))
        else:
            # Get current count of desired objects.
            self.cur_obj_count = len(env.world.get_all_object_locs(self.goal_obj))
            # Goal state is reached when the number of desired objects has increased.
            self.has_more_obj = lambda x: int(x) > self.cur_obj_count
            self.is_goal_state = lambda h: self.has_more_obj(
                    len(self.repr_to_env_dict[h].world.get_all_object_locs(self.goal_obj)))
            if self.removed_object is not None and self.removed_object == self.goal_obj:
                self.is_subtask_complete = lambda w: self.has_more_obj(
                        len(w.get_all_object_locs(self.goal_obj)) + 1)
            else:
                self.is_subtask_complete = lambda w: self.has_more_obj(
                        len(w.get_all_object_locs(self.goal_obj)))

    def _configure_planner_space(self, subtask_agent_names):
        """Configure planner to either plan in joint space or single-agent space."""
        assert len(subtask_agent_names) <= 2, "Cannot have more than 2 agents! Hm... {}".format(subtask_agents)

        self.is_joint = len(subtask_agent_names) == 2

    def set_settings(self, env, subtask, subtask_agent_names):
        """Configure planner."""
        # Configuring the planner level.
        # self._configure_planner_level(
        #         env=env,
        #         subtask_agent_names=subtask_agent_names,
        #         other_agent_planners=other_agent_planners)

        # Configuring subtask related information.
        # self._configure_subtask_information(
                # subtask=subtask,
                # subtask_agent_names=subtask_agent_names)

        # Defining what the goal is for this planner.
        # self._define_goal_state(
                # env=env,
                # subtask=subtask)

        # Defining the space of the planner (joint or single).
        # self._configure_planner_space(subtask_agent_names=subtask_agent_names)

        # Make sure termination counter has been reset.
        self.counter = 0
        self.num_explorations = 0
        self.stop = False
        self.num_explorations = 0

        # Set start state.
        self.start = copy.copy(env)
        self.repr_init(env_state=env)
        self.value_init(env_state=env)


    def repr_init(self, env_state):
        """Initialize repr for environment state."""
        es_repr = env_state.get_repr()
        if es_repr not in self.repr_to_env_dict:
            self.repr_to_env_dict[es_repr] = copy.copy(env_state)
        return es_repr

    def value_init(self, env_state):
        """Initialize value for environment state."""
        # Skip if already initialized.
        es_repr = env_state.get_repr()
        if ((es_repr, self.subtask) in self.v_l and
            (es_repr, self.subtask) in self.v_u):
            return

    def _get_modified_state_with_other_agent_actions(self, state):
        """Do nothing if the planner level is level 0.

        Otherwise, using self.other_agent_planners, anticipate what other agents will do
        and modify the state appropriately.

        Returns the modified state and the actions of other agents that triggered
        the change.
        """
        modified_state = copy.copy(state)
        other_agent_actions = {}

        # Do nothing if the planner level is 0.
        if self.planner_level == PlannerLevel.LEVEL0:
            return modified_state, other_agent_actions

        # Otherwise, modify the state because Level 1 planners
        # consider the actions of other agents.
        for other_agent_name, other_planner in self.other_agent_planners.items():
            # Keep their recipe subtask & subtask agent fixed, but change
            # their planner state to `state`.
            # These new planners should be level 0 planners.
            other_planner.set_settings(env=copy.copy(state),
                                       subtask=other_planner.subtask,
                                       subtask_agent_names=other_planner.subtask_agent_names)

            assert other_planner.planner_level == PlannerLevel.LEVEL0

            # Figure out what their most likely action is.
            possible_actions = other_planner.get_actions(state_repr=other_planner.start.get_repr())
            greedy_action = possible_actions[
                    argmin([other_planner.Q(state=other_planner.start,
                                            action=action,
                                            value_f=other_planner.v_l)
                    for action in possible_actions])]

            if other_planner.is_joint:
                greedy_action = greedy_action[other_planner.subtask_agent_names.index(other_agent_name)]

            # Keep track of their actions.
            other_agent_actions[other_agent_name] = greedy_action
            other_agent = list(filter(lambda a: a.name == other_agent_name,
                                      modified_state.sim_agents))[0]
            other_agent.action = greedy_action

        # Initialize state if it's new.
        self.repr_init(env_state=modified_state)
        self.value_init(env_state=modified_state)
        return modified_state, other_agent_actions

    def get_next_action(self, env, qtable):
        """Return next action."""
        #print("-------------[GET NEXT ACTION]-----------")
        
        # Configure planner settings.
        # self.set_settings(
        #         env=env
        #         )

        self.start = copy.copy(env)
        #self.repr_init(env_state=env)
        #self.value_init(env_state=env)

        # self.start is a copy of the environment.
        cur_state = copy.copy(self.start) #, _ = self._get_modified_state_with_other_agent_actions(state=self.start)

        #cur_state.display()
        cur_state_encoded = cur_state.encode()
        #print('current state encoded: ', cur_state_encoded)
        stateNr = int(cur_state_encoded, 2)
        # print("state nr: ", stateNr)

        action = np.argmax(qtable[stateNr,:])

        # action = env.world.get_action_name(action)
        

        # print("Action: ", action)
        

        return action


    # def getIdForState(self, cur_state_encoded):
    #     return np.sum(np.nonzero(cur_state_encoded))
        