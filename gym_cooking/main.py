# from environment import OvercookedEnvironment
# from gym_cooking.envs import OvercookedEnvironment
from recipe_planner.recipe import *
from utils.world import World
from utils.agent import RealAgent, SimAgent, COLORS
from utils.core import *
from misc.game.gameplay import GamePlay
from misc.metrics.metrics_bag import Bag

import numpy as np
import random
import argparse
from collections import namedtuple

import gym

#ACTION_TO_NAME = {(0, 1): 0, (0, -1): 1, (-1, 0): 2, (1, 0): 3} # (0, 0): 4}
ACTION_TO_NAME = {(0, -1): 0,  (-1, 0): 1, (1, 0): 2} # (0, 0): 4}

def parse_arguments():
    parser = argparse.ArgumentParser("Overcooked 2 argument parser")

    # Environment
    parser.add_argument("--level", type=str, required=True)
    parser.add_argument("--num-agents", type=int, required=True)
    parser.add_argument("--max-num-timesteps", type=int, default=500, help="Max number of timesteps to run")
    parser.add_argument("--max-num-subtasks", type=int, default=14, help="Max number of subtasks for recipe")
    parser.add_argument("--seed", type=int, default=1, help="Fix pseudorandom seed")
    parser.add_argument("--with-image-obs", action="store_true", default=False, help="Return observations as images (instead of objects)")
    parser.add_argument("--num_episodes", type=int, default=1000, help="Max number of episodes")
    parser.add_argument("--train", action="store_true", default=False, help="Whether or not we are running a trained agent")


    # Q-learning parameters
    parser.add_argument("--lr", type=float, default=0.6, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.8, help="gamma")
    parser.add_argument("--epsilon", type=float, default=1.0, help="epsilon")


    # Visualizations
    parser.add_argument("--play", action="store_true", default=False, help="Play interactive game with keys")
    parser.add_argument("--record", action="store_true", default=False, help="Save observation at each time step as an image in misc/game/record")

    return parser.parse_args()


def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def initialize_agents(arglist):
    real_agents = []

    with open('utils/levels/{}.txt'.format(arglist.level), 'r') as f:
        phase = 1
        recipes = []
        for line in f:
            line = line.strip('\n')
            if line == '':
                phase += 1

            # phase 2: read in recipe list
            elif phase == 1:
                recipes.append(globals()[line]())

            # phase 3: read in agent locations (up to num_agents)
            elif phase == 3:
                if len(real_agents) < arglist.num_agents:
                    loc = line.split(' ')
                    real_agent = RealAgent(
                            arglist=arglist,
                            name='agent-'+str(len(real_agents)+1),
                            id_color=COLORS[len(real_agents)],
                            recipes=recipes)
                    real_agents.append(real_agent)

    return real_agents

def read_recipe(arglist):
    with open('utils/levels/{}.txt'.format(arglist.level), 'r') as f:
        phase = 1
        recipes = []
        for line in f:
            line = line.strip('\n')
            if line == '':
                phase += 1

            # phase 2: read in recipe list
            elif phase == 1:
                recipes.append(globals()[line]())
    return recipes

def main_loop(arglist):
    """The main loop for running experiments."""
    print("Initializing environment and agents.")
    env = gym.envs.make("gym_cooking:overcookedEnv-v0", arglist=arglist)

    obs = env.reset()

    # Info bag for saving pkl files
    bag = Bag(arglist=arglist, filename=env.filename)
    # bag.set_recipe(recipe_subtasks=env.all_subtasks)

    # state represented as one-hot encoded location of agent (x and y) and 
    # hold tomato, hold plate, and chopped/not 2x2x2
    # four possible actions, moving left/up/right/down
    #((env.world.width*env.world.height)**2)*
    num_ingredients = 2
    
    qtable = np.zeros((((env.world.width*env.world.height)**(2+num_ingredients))*(2**(1+2*num_ingredients)), len(env.world.NAV_ACTIONS)))

    epsilon = arglist.epsilon
    max_epsilon = 1.0
    min_epsilon = 0.01
    decay_rate = 0.01
    thr = 0.0001
    qdiff = 100
    
    for episode in range(arglist.num_episodes):
        if qdiff > thr:
            total_reward = 0
            old_qtable = np.copy(qtable)

            print("-----------EPISODE %d-----------" % episode )
            print("===epsilon: %f ===" % epsilon)

            #obs contains the OvercookedEnvironment object
            #print("-----------RESET ENV -----------" )
            obs = env.reset()
            
            real_agents = initialize_agents(arglist=arglist)
            # Uncomment to print out the environment.
            #env.display()
            
            while not env.done():
                action_dict = {}


                for agent in real_agents:
                    agent.init_action(obs=obs)
                    action = agent.select_action(obs=obs, qtable=qtable, epsilon=epsilon)
                    action_dict[agent.name] = action
                new_obs, reward, _, info = env.step(action_dict=action_dict, episode=episode)
            
                # update Q-values

                for agent in real_agents:
                    action = action_dict[agent.name]
                    state = int(obs.encode(), 2)
                    new_state = int(new_obs.encode(), 2)
                    

                    qtable[state, action] = qtable[state, action] + arglist.lr * (reward + arglist.gamma * np.max(qtable[new_state, :]) - qtable[state, action])

                    agent.refresh_subtasks(world=env.world)
                qdiff = abs(qtable[state,action] - old_qtable[state,action])
                #print("qtable: ", qtable[idx])
                obs = new_obs
                print("reward: ", reward)
                
                total_reward += reward
                # Saving info
                bag.add_status(cur_time=info['t'], real_agents=real_agents, episode=episode)

            bag.set_termination(termination_info=env.termination_info,
                    successful=env.successful, total_r=total_reward, qtable=qtable, episode=episode)

            print("===total reward of episode %d %f === " % (episode, total_reward))
            print("QTABLE: \n")

            print(qtable)

            epsilon = min_epsilon + (max_epsilon - min_epsilon)* np.exp(-decay_rate*episode)
            
if __name__ == '__main__':
    arglist = parse_arguments()
    assert 0.0 <= arglist.gamma <= 1.0, "should be between 0.0 and 1.0"
    assert 0.0 <= arglist.epsilon <= 1.0, "should be between 0.0 and 1.0"
    assert 0.0 <= arglist.lr <= 1.0, "should be between 0.0 and 1.0"
 
    if arglist.play:
        env = gym.envs.make("gym_cooking:overcookedEnv-v0", arglist=arglist)
        env.reset()
        recipe = read_recipe(arglist)
        game = GamePlay(env.filename, env.world, env.sim_agents, recipe)
        game.on_execute()
    else:
        fix_seed(seed=arglist.seed)
        main_loop(arglist=arglist)