# Recipe planning
from recipe_planner.stripsworld import STRIPSWorld
import recipe_planner.utils as recipe_utils
from recipe_planner.utils import *
from delegation_planner.utils import SubtaskAllocDistribution

# Delegation planning
from delegation_planner.bayesian_delegator import BayesianDelegator

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
# CHANGE_MOVE_DOWN
ACTION_TO_NAME = {(0, 1): 0, (0, -1): 1, (-1, 0): 2, (1, 0): 3} # (0, 0): 4}
## REMOVED (0, 1)

class RealAgent:
    """Real Agent object that performs task inference and plans."""

    def __init__(self, arglist, name, id_color, recipes):
        self.arglist = arglist
        self.name = name
        self.color = id_color
        self.recipes = recipes

        # Bayesian Delegation.
        self.reset_subtasks()
        self.new_subtask = None
        self.new_subtask_agent_names = []
        self.incomplete_subtasks = []
        self.signal_reset_delegator = False
        self.is_subtask_complete = lambda w: False

        self.none_action_prob = 0.5

        self.planner = QLEARNING()


    def __str__(self):
        return color(self.name[-1], self.color)

    def __copy__(self):
        a = Agent(arglist=self.arglist,
                name=self.name,
                id_color=self.color,
                recipes=self.recipes)
        a.subtask = self.subtask
        a.new_subtask = self.new_subtask
        a.subtask_agent_names = self.subtask_agent_names
        a.new_subtask_agent_names = self.new_subtask_agent_names
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
        # if obs.t == 0:
        #     self.setup_subtasks(env=obs)

        # self.update_subtasks(env=obs)
        # self.new_subtask, self.new_subtask_agent_names = self.select_subtask()
        # print("subtask: ", self.new_subtask)
        # print("===Qtable==: ", qtable)

        rand_nr = random.uniform(0,1)
        if rand_nr > epsilon:
            self.plan(copy.copy(obs),copy.copy(qtable))
        else:
            # Check whether this subtask is done.
            # self.def_subtask_completion(env=obs)
            self.action = ACTION_TO_NAME[random.choice(obs.world.NAV_ACTIONS)]
            #print("self.action: ", ACTION_TO_WORD[self.action])
               #np.random.choice(env.world.NAV_ACTIONS)
        
        return self.action

    def select_subtask(self):
        max_subtask_alloc = self.add_greedy_subtasks().get_max()
        if max_subtask_alloc is not None:
            for t in max_subtask_alloc:
                if self.name in t.subtask_agent_names:
                    return t.subtask, t.subtask_agent_names
        return None, self.name       

    def add_greedy_subtasks(self):
        """Return the entire distribution of greedy subtask allocations.
        i.e. subtasks performed only by agent with self.agent_name."""
        subtask_allocs = []

        subtasks = self.incomplete_subtasks
        # At least 1 agent must be doing something.
        if None not in subtasks:
            subtasks += [None]

        # Assign this agent to all subtasks. No joint subtasks because this function
        # only considers greedy subtask allocations.
        for subtask in subtasks:
            subtask_alloc = [SubtaskAllocation(subtask=subtask, subtask_agent_names=(self.name,))]
            subtask_allocs.append(subtask_alloc)
        return SubtaskAllocDistribution(subtask_allocs)

    def get_subtasks(self, world):
        """Return different subtask permutations for recipes."""
        self.sw = STRIPSWorld(world, self.recipes)
        # [path for recipe 1, path for recipe 2, ...] where each path is a list of actions.
        subtasks = self.sw.get_subtasks(max_path_length=self.arglist.max_num_subtasks)
        all_subtasks = [subtask for path in subtasks for subtask in path]

        # Uncomment below to view graph for recipe path i
        # i = 0
        # pg = recipe_utils.make_predicate_graph(self.sw.initial, recipe_paths[i])
        # ag = recipe_utils.make_action_graph(self.sw.initial, recipe_paths[i])
        #print("All subtasks: ", all_subtasks)
        return all_subtasks

    def setup_subtasks(self, env):
        """Initializing subtasks and subtask allocator, Bayesian Delegation."""
        self.incomplete_subtasks = self.get_subtasks(world=env.world)
        # self.delegator = BayesianDelegator(
        #         agent_name=self.name,
        #         all_agent_names=env.get_agent_names(),
        #         model_type='greedy',
        #         planner=self.planner,
        #         none_action_prob=self.none_action_prob)

    def reset_subtasks(self):
        """Reset subtasks---relevant for Bayesian Delegation."""
        self.subtask = None
        self.subtask_agent_names = []
        self.subtask_complete = False

    def refresh_subtasks(self, world):
        # Check whether subtask is complete.
        self.subtask_complete = False
        #if self.subtask is None or len(self.subtask_agent_names) == 0:
            #print("{} has no subtask".format(color(self.name, self.color)))
            #return
        self.subtask_complete = self.is_subtask_complete(world)

        #print("{} done with {} according to planner: {}\n".format(color(self.name, self.color), self.subtask, self.is_subtask_complete(world)))

        # Refresh for incomplete subtasks.
        if self.subtask_complete:
            if self.subtask in self.incomplete_subtasks:
                self.incomplete_subtasks.remove(self.subtask)
                self.subtask_complete = True
        # print('{} incomplete subtasks:'.format(
        #     color(self.name, self.color)),
        #     ', '.join(str(t) for t in self.incomplete_subtasks))

    # def update_subtasks(self, env):
    #     """Update incomplete subtasks---relevant for Bayesian Delegation."""
    #     # if ((self.subtask is not None and self.subtask not in self.incomplete_subtasks):
    #     #         or (self.delegator.should_reset_priors(obs=copy.copy(env),
    #     #                     incomplete_subtasks=self.incomplete_subtasks))):
    #     #     self.reset_subtasks()
    #     #     self.delegator.set_priors(
    #     #             obs=copy.copy(env),
    #     #             incomplete_subtasks=self.incomplete_subtasks,
    #     #             priors_type=self.priors)
    #     # else:
    #     #     if self.subtask is None:
    #     #         self.delegator.set_priors(
    #     #             obs=copy.copy(env),
    #     #             incomplete_subtasks=self.incomplete_subtasks,
    #     #             priors_type=self.priors)
    #     #     else:
    #     #         self.delegator.bayes_update(
    #     #                 obs_tm1=copy.copy(env.obs_tm1),
    #     #                 actions_tm1=env.agent_actions,
    #     #                 beta=self.beta)
    #     self.reset_subtasks()

    def all_done(self):
        """Return whether this agent is all done.
        An agent is done if all Deliver subtasks are completed."""
        if any([isinstance(t, Deliver) for t in self.incomplete_subtasks]):
            return False
        return True

    def get_action_location(self):
        """Return location if agent takes its action---relevant for navigation planner."""
        return tuple(np.asarray(self.location) + np.asarray(self.action))

    def plan(self, env, qtable, initializing_priors=False):
        """Plan next action---relevant for navigation planner."""
        #print('right before planning, {} had old subtask {}, new subtask {}, subtask complete {}'.format(self.name, self.subtask, self.new_subtask, self.subtask_complete))
        print('agent is taking action from qtable')
        # Check whether this subtask is done.
        #self.def_subtask_completion(env=env)

        # If subtask is None, then do nothing. ????
        # if (self.new_subtask is None) or (not self.new_subtask_agent_names):
        #     actions = nav_utils.get_single_actions(env=env, agent=self)
        #     probs = []
        #     for a in actions:
        #         if a == (0, 0):
        #             probs.append(self.none_action_prob)
        #         else:
        #             probs.append((1.0-self.none_action_prob)/(len(actions)-1))
        #     self.action = actions[np.random.choice(len(actions), p=probs)]
        # Otherwise, plan accordingly.
        # else:
        #other_agent_planners = {}
        
        #print("[ {} Planning ] Task: {}, Task Agents: {}".format(
        #    self.name, self.new_subtask, self.new_subtask_agent_names))

        # get argmax, biggest q value for this state.

        self.action = self.planner.get_next_action(
                env=env, qtable=qtable)

        # Update subtask.
        # self.subtask = self.new_subtask
        # self.subtask_agent_names = self.new_subtask_agent_names
        # self.new_subtask = None
        # self.new_subtask_agent_names = []

        # print('{} proposed action: {}\n'.format(self.name, self.action))

    def def_subtask_completion(self, env):
        # Determine desired objects.
        self.start_obj, self.goal_obj = nav_utils.get_subtask_obj(subtask=self.new_subtask)
        self.subtask_action_object = nav_utils.get_subtask_action_obj(subtask=self.new_subtask)

        # Define termination conditions for agent subtask.
        # For Deliver subtask, desired object should be at a Deliver location.
        for subtask in env.run_recipes():
            #print("SUBTASK IN COMPLETION CHECK: ", subtask)
            if self.start_obj and self.goal_obj:
                if isinstance(subtask, Deliver):
                    self.cur_obj_count = len(list(
                        filter(lambda o: o in set(env.world.get_all_object_locs(self.subtask_action_object)),
                        env.world.get_object_locs(obj=self.goal_obj, is_held=False))))
                    self.has_more_obj = lambda x: int(x) > self.cur_obj_count
                    self.is_subtask_complete = lambda w: self.has_more_obj(
                            len(list(filter(lambda o: o in
                        set(env.world.get_all_object_locs(obj=self.subtask_action_object)),
                        w.get_object_locs(obj=self.goal_obj, is_held=False)))))
                # Otherwise, for other subtasks, check based on # of objects.
                else:
                    # Current count of desired objects.
                    self.cur_obj_count = len(env.world.get_all_object_locs(obj=self.goal_obj))
                    # Goal state is reached when the number of desired objects has increased.
                    print(w.get_all_object_locs(obj=self.goal_obj))
                    print(self.cur_obj_count)
                    print(lambda w: len(w.get_all_object_locs(obj=self.goal_obj)) > self.cur_obj_count)
                    quit()
                    self.is_subtask_complete = lambda w: len(w.get_all_object_locs(obj=self.goal_obj)) > self.cur_obj_count


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
