'''
Determine optimal policy for Group 1 -> i.e. most successful action overall.
'''
import numpy as np
import pandas as pd
import pickle

data  = pd.read_csv('data_samples_post_sess_2.csv', converters={'s0': eval, 's1': eval})
data = data.values.tolist()
num_samples = len(data)
num_act = 4

rewards = np.zeros(4)
trials = np.zeros(4)

for data_index in range(num_samples):
    rewards[data[data_index][2]] += data[data_index][3]
    trials[data[data_index][2]] += 1

# Calculate success rate per action
success_rate = np.divide(rewards, trials,
                         out = np.zeros_like(rewards),
                         where=trials!=0)

# Get as optimal policy the actions with highest success rates
opt_policy = [i for i in range(num_act) if success_rate[i] == max(success_rate)]

with open('Level_1_Optimal_Policy', 'wb') as f:
    pickle.dump(opt_policy, f)