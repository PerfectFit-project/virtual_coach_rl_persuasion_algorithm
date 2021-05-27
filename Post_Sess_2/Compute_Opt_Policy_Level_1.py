'''
Determine optimal policy for Group 1 -> i.e. most successful action overall.
'''
import numpy as np
import pandas as pd
import pickle
import Utils as util

# Load data extracted from database
# TODO: use correct path
data  = pd.read_csv('W:/staff-umbrella/perfectfit/Exp0/Final_Algorithms/2021_05_27_1401_data_samples_post_sess_2.csv', converters={'s0': eval, 's1': eval})
data = data.values.tolist()
num_samples = len(data)

# All effort responses
list_of_efforts = list(np.array(data)[:, 3].astype(int))
# Mean value of effort responses
# TODO: use correct path
with open("W:/staff-umbrella/perfectfit/Exp0/Final_Algorithms/2021_05_27_1401_Post_Sess_2_Effort_Mean", "rb") as f:
    effort_mean = pickle.load(f)
# Map effort responses to rewards from 0 to 1, with the mean mapped to 0.5.
map_to_rewards = util.get_map_effort_reward(effort_mean, output_lower_bound = 0, 
                                            output_upper_bound = 1, 
                                            input_lower_bound = 0, 
                                            input_upper_bound = 10)
reward_list = util.map_efforts_to_rewards(list_of_efforts, map_to_rewards)

# number of actions
# we have 5, 4 actions that persuade, and the last action is to
# not persuade people
num_act = 5

rewards = np.zeros(num_act)
trials = np.zeros(num_act)

for data_index in range(num_samples):
    rewards[data[data_index][2]] += reward_list[data_index]
    trials[data[data_index][2]] += 1

# Calculate average reward per action
avg_reward = np.divide(rewards, trials,
                       out = np.zeros_like(rewards),
                       where=trials!=0)

# Get as optimal policy the actions with highest average reward
opt_policy = [i for i in range(num_act) if avg_reward[i] == max(avg_reward)]

with open('W:/staff-umbrella/perfectfit/Exp0/Final_Algorithms/Level_1_Optimal_Policy', 'wb') as f:
    pickle.dump(opt_policy, f)