'''
Feature selection based on success rates, this time with a wrapper approach.
To be run after session 2.
'''
import numpy as np
import itertools
from scipy import stats
import pickle
import Utils as util
import pandas as pd
import random
from sklearn.linear_model import LinearRegression

# load data. Data has <s, s', a, r> samples.
feat_to_select = [0, 1, 2, 3, 4, 6, 7]
data  = pd.read_csv('data_samples_post_sess_2.csv', converters={'s0': eval, 's1': eval})
data = data.values.tolist()

# All effort responses
list_of_efforts = list(np.array(data)[:, 3].astype(int))
# Mean value of effort responses
with open("Post_Sess_2_Effort_Mean", "rb") as f:
    effort_mean = pickle.load(f)
# Map effort responses to rewards from 0 to 1, with the mean mapped to 0.5.
map_to_rewards = util.get_map_effort_reward(effort_mean, output_lower_bound = 0, 
                                            output_upper_bound = 1, 
                                            input_lower_bound = 0, 
                                            input_upper_bound = 10)
reward_list = util.map_efforts_to_rewards(list_of_efforts, map_to_rewards)
# now write these obtained reward values into "data"
for i in range(len(reward_list)):
    data[i][3] = reward_list[i]

num_act = 5 # number of actions
num_feat = len(feat_to_select)
num_samples = len(data)

num_states = 2 ** num_feat
actions = [i for i in range(num_act)]

all_states = list(map(list, itertools.product([0, 1], repeat = num_feat)))

# first we need to get the data for each state
rewards = np.zeros((2 ** num_feat, num_act))
trials = np.zeros((2 ** num_feat, num_act))
success_rates = np.zeros((2 ** num_feat, num_act))
for data_index in range(num_samples): # for each data sample
    s = data[data_index][0]
    a = data[data_index][2]
    r = data[data_index][3]
    
    state_idx = all_states.index(s)
    rewards[state_idx, a] += r
    trials[state_idx, a] += 1
  
# Now we compute the success rates for each state.
df_list = []
for state_idx, state in enumerate(all_states):
    for a in range(num_act):
        success_rate = 0
        if trials[state_idx, a] > 0:
            success_rate = rewards[state_idx][a]/trials[state_idx][a]
        # If an action has not been tried, we assign an avg reward of 0.5
        # This means that in absence of any observations, we expect the outcome of an action to be 
        # the mean effort response
        else:
            success_rate = 0.5
        success_rates[state_idx][a] = success_rate

# Create a dataframe that contains all features, actions, and success rates
df_list = []
for state_idx, state in enumerate(all_states):
    for a in range(num_act):
        df_list.append(state + [a] + [success_rates[state_idx][a]])
df_all = pd.DataFrame(df_list)

num_feat_to_select = 3 # number of features to select
abstract_states = list(map(list, itertools.product([0, 1], repeat = num_feat_to_select)))

# all sets of features that we could select
all_sets = [list(i) for i in itertools.combinations([s for s in range(num_feat)], num_feat_to_select)]

# for each of these sets we now need to do 10-fold cross validation
for set_idx, sett in enumerate(all_sets):
    linear_regressor = LinearRegression()  # create object for the class
    
    linear_regressor.fit(X, Y)

# Store selected features
with open('Level_2_G_algorithm_chosen_features', 'wb') as f:
    pickle.dump(feat_sel, f)
with open("Level_2_G_algorithm_chosen_features_criteria", 'wb') as f:
    pickle.dump(feat_sel_criteria, f)
    
# Compute best action per state in chosen representation
total_reward = np.zeros((2, 2, 2, num_act))
total_trial = np.zeros((2, 2, 2, num_act))

for data_index in range(num_samples): # for each data sample
    s_b = np.take(data[data_index][0], feat_sel)
    index = list(s_b) + [data[data_index][2]]
    total_reward[index[0], index[1], index[2], index[3]] += data[data_index][3]
    total_trial[index[0], index[1], index[2], index[3]] += 1
    
success_rates = np.divide(total_reward, total_trial,
                          out = np.zeros_like(total_reward),
                          where=total_trial!=0)

# If we did not try an action in a state, we assume a default success rate of 0.5, which is halfway between 0 and 1.
success_rates = np.array([[[[success_rates[i][j][k][l] if total_trial[i][j][k][l] > 0 else 0.5 for l in range(num_act)] for k in range(2)] for j in range(2)] for i in range(2)])

optimal_policy = [[[[a for a in range(num_act) if success_rates[i, j, k, a] == max(success_rates[i, j, k])] for k in range(2)] for j in range(2)] for i in range(2)]

with open('Level_2_Optimal_Policy', 'wb') as f:
    pickle.dump(optimal_policy, f)