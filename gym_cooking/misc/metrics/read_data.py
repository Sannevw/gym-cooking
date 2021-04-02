import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import pickle
import sys
sys.path.append("../..")
import recipe_planner


recipes = [
        "tomato",
    ]
total_num_subtasks = {
        "tomato": 3,
    }
maps = [
        "open-divider",
        ]
seeds = [1]
agents = ['agent-1']

ylims = {
    'time_steps': [0, 500],
    'total_rewards': [-5000, 5000]
}

ylabels = {
    'time_steps': 'Time',
    'total_rewards': 'Total Rewards'
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="for parsing")
    parser.add_argument("--num-agents", type=int, default=1, help="number of agents")
    parser.add_argument("--stats", action="store_true", default=False, help="run and print out summary statistics")
    parser.add_argument("--time-steps", action="store_true", default=False, help="make graphs for time_steps")
    parser.add_argument("--legend", action="store_true", default=False, help="make legend alongside graphs")
    return parser.parse_args()


def run_main():
    #path_pickles = '/Users/custom/path/to/pickles'
    path_pickles = os.path.join(os.getcwd(), 'pickles/today')
    #path_save = '/Users/custom/path/to/save/to'
    path_save = os.path.join(os.getcwd(), 'reward_graphs_agents{}'.format(arglist.num_agents))
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    data = import_data(path_pickles, arglist.num_agents)
    print('done loading pickle data')
    print("DF: ", data)
    plot_rewards(data)
    quit()
    #plot_data(key, path_save, df, arglist.num_agents, legend=arglist.legend)

def import_data(path_pickles, num_agents):
    # df = list()

    # for recipe, map_, seed in itertools.product(recipes, maps, seeds):
    #     info = {
    #         "map": map_,
    #         "seed": seed,
    #         "recipe": recipe,
    #         "dummy": 0
    #     }

    # LOAD IN FILE
    fname = 'open-divider_tomato_agents1_seed1.pkl'
    print("path: ", path_pickles) #'{}_{}_agents{}_seed{}{}.pkl'.format(map_, recipe, num_agents, seed, model)
    if os.path.exists(os.path.join(path_pickles, fname)):
        try:
            data = pickle.load(open(os.path.join(path_pickles, fname), "rb"))
            print("===try loading data===")
        except:
            print("trouble loading: {}".format(fname))
    return data

def plot_rewards(data):
    rewards = []
    xaxis = []

    for i, episode in enumerate(data.keys()):
        # TOTAL REWARDS
        tr = data[i]['total_rewards']
        rewards.append(tr)
        xaxis.append(i+1)
        #df.append(dict({'episode': i, 'tr': tr}, **info))
            
    fig = plt.figure()
    ax = plt.axes()
    plt.plot(xaxis, rewards)
    plt.show()


def compute_stats(path_pickles, num_agents):
    for model in models:
        num_shuffles = []; num_timesteps = []; num_collisions = []; frac_completed = []
        for recipe, map_, seed in itertools.product(recipes, maps, seeds):
            fname = '{}_{}_agents{}_seed{}{}.pkl'.format(map_, recipe, num_agents, seed, model)
            if os.path.exists(os.path.join(path_pickles, fname)):
                try:
                    data = pickle.load(open(os.path.join(path_pickles, fname), "rb"))
                except EOFError:
                    continue
                shuffles = get_shuffles(data, recipe)   # dict of 2 numbers
                num_shuffles += shuffles.values()
                num_timesteps.append(get_time_steps(data, recipe))
                if data['was_successful']:
                    num_collisions.append(get_collisions(data, recipe))
                frac_completed.append(get_frac_completed(data, recipe))
            else:
                print('no file:', fname)
                continue

        print('{}   time steps: {:.3f} +/- {:.3f}'.format(model_key[model], np.mean(np.array(num_timesteps)), np.std(np.array(num_timesteps))/np.sqrt(len(num_timesteps))))
        print('     frac_completed: {:.3f} +/- {:.3f}'.format(np.mean(np.array(frac_completed)), np.std(np.array(num_collisions))/np.sqrt(len(frac_completed))))
        print('     collisions: {:.3f} +/- {:.3f}'.format(np.mean(np.array(num_collisions)), np.std(np.array(num_collisions))/np.sqrt(len(num_collisions))))
        print('     shuffles: {:.3f} +/- {:.3f}'.format(np.mean(np.array(num_shuffles)), np.std(np.array(num_collisions))/np.sqrt(len(num_shuffles))))



        # COMPLETION
        # elif key == 'completion':
        #     for t in range(100):
        #         n = get_completion(data, recipe, t)
        #         df.append(dict({'t': t-1, 'n': n}, **info))

        # SHUFFLES
        # elif key == 'shuffles':
        #     shuffles = get_shuffles(data, recipe)   # a dict
        #     df.append(dict(shuffles = np.mean(np.array(list(shuffles.values()))), **info))
            # for agent in agents:
            #     info['agent'] = agent
            #     df.append(dict(shuffles = shuffles[agent], **info))

    return pd.DataFrame(df)

def get_time_steps(data, recipe):
    try:
        # first timestep at which required number of recipe subtasks has been completed
        # using this instead of total length bc of termination bug
        return data['num_completed_subtasks'].index(total_num_subtasks[recipe])+1
    except:
        return 100

def get_completion(data, recipe, t):
    df = list()
    completion = data['num_completed_subtasks']
    try:
        end_indx = completion.index(total_num_subtasks[recipe])+1
        completion = completion[:end_indx]
    except:
        end_indx = None
    if len(completion) < 100:
        completion += [data['num_completed_subtasks_end']]*(100-len(completion))
    assert len(completion) == 100
    return completion[t]/total_num_subtasks[recipe]

def get_shuffles(data, recipe):
    # recipe isn't needed but just for consistency
    # returns a dict, although we only use average of all agents
    shuffles = {}
    for agent in data['actions'].keys():
        count = 0
        actions = data['actions'][agent]
        holdings = data['holding'][agent]
        for t in range(2, len(holdings)):
            # count how many negated the previous action
            # redundant movement
            if holdings[t-2] == holdings[t-1] and holdings[t-1] == holdings[t]:
                net_action = np.array(actions[t-1]) + np.array(actions[t])
                redundant = (net_action == [0, 0])
                if redundant.all() and actions[t] != (0, 0):
                    count += 1
                    # print(agent, t, actions[t-1], holdings[t-1], actions[t], holdings[t], actions[t+1], holdings[t+1])
            # redundant interaction
            elif holdings[t-2] != holdings[t-1] and holdings[t-2] == holdings[t]:
                redundant = (actions[t-1] == actions[t] and actions[t] != (0, 0))
                if redundant:
                    count += 1
                    # print(agent, t, actions[t-1], holdings[t-1], actions[t], holdings[t], actions[t+1], holdings[t+1])
        shuffles[agent] = count
    return shuffles

def plot_data(key, path_save, df, num_agents, legend=False):
    print('generating {} graphs'.format(key))
    hue_order = [model_key[l] for l in models]
    color_palette = sns.color_palette()
    sns.set_style('ticks')
    sns.set_context('talk', font_scale=1)
    print("df: ", df)

    for i, recipe in enumerate(recipes):
        for j, map_ in enumerate(maps):
            try:
                data = df.loc[(df['map']==map_) & (df['recipe']==recipe), :]
            except:
                continue
            if len(data) == 0:
                print('empty data on ', (recipe, map_))
                continue

            plt.figure(figsize=(3,3))

            if key == 'completion':
                # plot ours last
                hue_order = hue_order[1:] + [hue_order[0]]
                color_palette = sns.color_palette()[1:5] + [sns.color_palette()[0]]

                ax = sns.lineplot(x = 't', y = 'n', hue="model", data=data,
                    linewidth=5, legend=False, hue_order=hue_order, palette=color_palette, n_colors=3)
                plt.xlabel('Steps')
                plt.ylim([0, 1]),
                plt.xlim([0, 100])
            else:
                hue_order = hue_order[1:] + [hue_order[0]]
                color_palette = sns.color_palette()[1:5] + [sns.color_palette()[0]]
                sns.barplot(x='dummy', y=key, hue="model", data=data, hue_order=hue_order,\
                                palette=color_palette, ci=68).set(
                    xlabel = "",
                    xticks = [],
                    ylim = ylims[key],
                )
            plt.legend('')
            plt.gca().legend().set_visible(False)
            sns.despine()
            plt.tight_layout()

            plt.ylabel(ylabels[key])

            if recipe != 'tomato' and key == 'priors':
                plt.gca().get_yaxis().set_visible(False)
                plt.gca().spines['left'].set_visible(False)
                plt.ylabel('')

            if key == 'time_steps' or key == 'priors':
                plt.axhline(y = time_steps_optimal[num_agents][map_][recipe], ls='--', color='black')

            plt.savefig(os.path.join(path_save, "{}_{}_{}.png".format(key, recipe, map_)))
            plt.close()

            print('   generated graph for {}, {}'.format(recipe, map_))

    # Make Legend
    if arglist.legend:
        plt.figure(figsize=(10,10))
        if key == 'completion':
            sns.barplot(x = 't', y = 'n', hue="model", data=data, hue_order=hue_order, palette=color_palette, ci=68).set()
        else:
            sns.barplot(x='dummy', y=key, hue="model", data=data, hue_order=hue_order, palette=color_palette, ci=68).set(
                xlabel = "", xticks = [], ylim = [0, 1000])
        legend = plt.legend(frameon=False)
        legend_fig = legend.figure
        legend_fig.canvas.draw()
        # bbox = legend.get_window_extent().transformed(legend_fig.dpi_scale_trans.inverted())
        # legend_fig.savefig(os.path.join(path_save, 'legend.pdf'), dpi="figure", bbox_inches=bbox)
        legend_fig.savefig(os.path.join(path_save, '{}_legend_full.png'.format(key)), dpi="figure")
        plt.close()



if __name__ == "__main__":
    arglist = parse_arguments()
    run_main()