import dill as pickle
import copy

class Bag:
    def __init__(self, arglist, filename):
        self.data = {}
        self.arglist = arglist
        self.directory = "misc/metrics/pickles/today/"
        self.filename = filename
        self.set_general()

    def set_general(self):
        # self.data["level"] = self.arglist.level
        # self.data["num_agents"] = self.arglist.num_agents

        # Prepare for agent information
        for episode in range(self.arglist.num_episodes):
            self.data[episode] = {}
            for info in ["states","actions", "holding"]:
                self.data[episode][info] = {"agent-{}".format(i+1): [] for i in range(self.arglist.num_agents)}

    def set_recipe(self, recipe_subtasks):
        self.data["all_subtasks"] = recipe_subtasks
        self.data["num_total_subtasks"] = len(recipe_subtasks)

    def add_status(self, cur_time, real_agents, episode):
        for a in real_agents:
            self.data[episode]["states"][a.name].append(copy.copy(a.location))
            self.data[episode]["holding"][a.name].append(a.get_holding())
            self.data[episode]["actions"][a.name].append(a.action)

    def set_termination(self, termination_info, successful, total_r, qtable, episode):
        self.data[episode]["termination"] = termination_info
        self.data[episode]["was_successful"] = successful
        self.data[episode]["total_rewards"] = total_r
        print("self.data.episode['total_rewards']: ", self.data[episode]["total_rewards"])
        self.data[episode]["qtable"] = qtable


        # print("Self data final: ", self.data)

        pickle.dump(self.data, open(self.directory+self.filename+'.pkl', "wb"))
        # print("episode: ", episode)
        # print("total rewards:" , total_r)
        # print("Saved to {}".format(self.directory+self.filename+'.pkl'))
