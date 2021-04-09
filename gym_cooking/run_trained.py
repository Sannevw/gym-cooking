from recipe_planner.recipe import *
from utils.world import World
from utils.agent_svw import RealAgent, SimAgent, COLORS
from utils.core import *
from misc.game.gameplay import GamePlay
# from misc.metrics.metrics_bag_svw import Bag

import numpy as np
import random
import argparse
from collections import namedtuple
import sys
import os
import pickle

import gym

def parse_arguments():
    parser = argparse.ArgumentParser("Overcooked argument parser")
    parser.add_argument("--fname", type=str, required=False)
    parser.add_argument("--data_dir", type=str, required=False)
    parser.add_argument("--train", action="store_true", default=True, help="Save observation at each time step as an image in misc/game/record")
    
    parser.add_argument("--level", type=str, required=True)
    parser.add_argument("--num-agents", type=int, required=True)
    parser.add_argument("--max-num-timesteps", type=int, default=300, help="Max number of timesteps to run")
    parser.add_argument("--max-num-subtasks", type=int, default=14, help="Max number of subtasks for recipe")
    parser.add_argument("--seed", type=int, default=1, help="Fix pseudorandom seed")
    parser.add_argument("--with-image-obs", action="store_true", default=False, help="Return observations as images (instead of objects)")
    parser.add_argument("--num_episodes", type=int, default=36, help="Max number of episodes")

 # Q-learning parameters
    parser.add_argument("--lr", type=float, default=0.4, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.7, help="gamma")
    parser.add_argument("--epsilon", type=float, default=1.0, help="epsilon")


    # Visualizations
    parser.add_argument("--play", action="store_true", default=False, help="Play interactive game with keys")
    parser.add_argument("--record", action="store_true", default=False, help="Save observation at each time step as an image in misc/game/record")

    return parser.parse_args()

def load_data(arglist):
    if arglist.data_dir:
        data_dir = arglist.data_dir
    else:
        data_dir = 'misc/metrics/pickles/today'
    if arglist.fname:
        fname = arglist.fname
    else:
        fname = 'open-divider_tomato_agents1_seed1.pkl'

    if os.path.exists(os.path.join(data_dir, fname)):
        try:
            print("Successfully loaded: {}".format(fname))
            data = pickle.load(open(os.path.join(data_dir, fname), "rb"))
        except:
            print("trouble loading: {}".format(fname))

    data_final = []
    for item in data.items():
        if 'qtable' in item[1].keys():
            data_final.append(item)
    return data_final

def get_qtable(data):
    # convert dict to list to obtain last entry
    # to get the final qtable, after last updates in training
    #dl = list(data.items())

    qtable = data[-1][1]['qtable']
    return qtable
    

ACTION_TO_NAME = {0: 'down', 1: 'up', 2: 'left', 3: 'right'} # (0, 0): 4}

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
            elif phase == 2:
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

if __name__ == '__main__':
    arglist = parse_arguments()

    print("=== RUN TRAINED AGENT AND CREATE VISUALIZATION ===")
    data = load_data(arglist)
    qtable = get_qtable(data)

    print("===Initializing environment and agents.===")
    env = gym.envs.make("gym_cooking:overcookedEnv-v0", arglist=arglist)

    obs = env.reset()
    max_steps = 100

    for episode in range(5):
        obs = env.reset()
        real_agents = initialize_agents(arglist=arglist)
        # color robot

        step_ = 0
        print("===Episode: %d ===" % episode)
        
        while not env.done() and step_ < max_steps:
            state = int(obs.encode(), 2)
            print("state: ", state)
            
            action_dict = {}
            print("qtable[state] ", qtable[state,:])
            print("state: ", state)
            for agent in real_agents:
                action = np.argmax(qtable[state,:])
                action_dict[agent.name] = action
            
            new_obs, reward, _, info = env.step(action_dict=action_dict, episode=episode)

            agent.refresh_subtasks(world=env.world)
            obs = new_obs

            step_ +=1
        if env.done():
            break


        






    
    
