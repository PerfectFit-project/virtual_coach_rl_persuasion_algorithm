'''
Feature selection as in G-algorithm but with success rates.
To be run after session 2.
We use a one-sample t-test.
This time we comopute the t-test based on all blocks at once.
'''
import numpy as np
import itertools
from scipy import stats
import pickle
import Utils as util
import pandas as pd
import random

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

states = list(map(list, itertools.product([0, 1], repeat = num_feat)))

num_feat_to_select = 3 # number of features to select
        
# Select first feature
rewards = np.zeros((num_feat, 2, num_act))
trials = np.zeros((num_feat, 2, num_act))
t_tests = np.zeros((num_feat))
for f in range(num_feat):
   for data_index in range(num_samples): # for each data sample
        s = data[data_index][0][f]
        a = data[data_index][2]
        r = data[data_index][3]
        
        rewards[f][s][a] += r
        trials[f][s][a] += 1
    
   # Compute success rate or rather average reward
   success_rate_0_ff = np.divide(rewards[f][0], trials[f][0], out=np.zeros_like(rewards[f][0]), where=trials[f][0]!=0)
   success_rate_1_ff = np.divide(rewards[f][1], trials[f][1], out=np.zeros_like(rewards[f][1]), where=trials[f][1]!=0)
   
   # If an action has not been tried, we assign an avg reward of 0.5
   # This means that in absence of any observations, we expect the outcome of an action to be 
   # the mean effort response
   success_rate_0_ff = [success_rate_0_ff[sr0_idx] if trials[f][0][sr0_idx] > 0 else 0.5 for sr0_idx in range(num_act)]
   success_rate_1_ff = [success_rate_1_ff[sr1_idx] if trials[f][1][sr1_idx] > 0 else 0.5 for sr1_idx in range(num_act)]
   
   # One-sample t-test, comparing to a difference of 0
   t_tests[f] = stats.ttest_1samp(np.abs(np.array(success_rate_0_ff) - np.array(success_rate_1_ff)), 0)[1]

# minimum p-value for t-test, ignoring nan-values
# we always have the same number of observations, so a larger effect size
# corresponds to a lower p-value. So we do not need to consider both
# the effect size and the p-value separately somehow.
min_p_val = np.nanmin(t_tests)
feat_sel_options = [i for i in range(num_feat) if t_tests[i] == min_p_val]
feat_sel = [random.choice(feat_sel_options)] # choose randomly if there are multiple best features
criterion = "First feature -> min. p-value: " + str(round(min_p_val, 4))
if len(feat_sel_options) > 1:
    criterion += " random from " + str(feat_sel_options)
feat_sel_criteria = [criterion]
print("First feature selected:", feat_sel[0], criterion)

# Select remaining features
for j in range(num_feat_to_select - 1):
    num_feat_sel = len(feat_sel)
    feat_not_sel = [f for f in range(num_feat) if not f in feat_sel]
    num_feat_not_sel = len(feat_not_sel)
    
    num_blocks = 2 ** num_feat_sel
    blocks = [list(i) for i in itertools.product([0, 1], repeat = num_feat_sel)]
    rewards_2 = np.zeros((num_blocks, num_feat_not_sel, 2, num_act))
    trials_2 = np.zeros((num_blocks, num_feat_not_sel, 2, num_act))
    t_tests_2 = np.zeros(num_feat_not_sel)
    
    # Compute t-test based on success-rates
    for f_ind, f in enumerate(feat_not_sel): # for each not yet selected feature
    
        success_rate_all_blocks_0 = []
        success_rate_all_blocks_1 = []
        
        for b_ind, block in enumerate(blocks): # for each block
            for data_index in range(num_samples): # for each data sample
                
                s = data[data_index][0][f]
                s_b = np.take(data[data_index][0], feat_sel)
                
                # both s must be in the current block
                if list(s_b) == block:
                    
                    a = data[data_index][2]
                    r = data[data_index][3]
                               
                    # save reward
                    rewards_2[b_ind][f_ind][s][a] += r
                    trials_2[b_ind][f_ind][s][a] += 1
                    
            
            success_rate_0  = np.divide(rewards_2[b_ind][f_ind][0], trials_2[b_ind][f_ind][0],
                                        out = np.zeros_like(rewards_2[b_ind][f_ind][0]),
                                        where=trials_2[b_ind][f_ind][0]!=0)
            success_rate_1  = np.divide(rewards_2[b_ind][f_ind][1], trials_2[b_ind][f_ind][1],
                                        out = np.zeros_like(rewards_2[b_ind][f_ind][1]),
                                        where=trials_2[b_ind][f_ind][1]!=0)
            
            # If an action has not been tried, we assign an avg reward of 0.5
            # This means that in absence of any observations, we expect the outcome of an action to be 
            # the mean effort response
            success_rate_0 = [success_rate_0[sr0_idx] if trials_2[b_ind][f_ind][0][sr0_idx] > 0 else 0.5 for sr0_idx in range(num_act)]
            success_rate_1 = [success_rate_1[sr1_idx] if trials_2[b_ind][f_ind][1][sr1_idx] > 0 else 0.5 for sr1_idx in range(num_act)]
            
            success_rate_all_blocks_0 += success_rate_0
            success_rate_all_blocks_1 += success_rate_1
            
        # t-test
        # one-sample: compare difference to a mean of 0
        # we take the absolute value of the difference
        # Otherwise, comparing values [0, -1, 1] to the mean 0 leads to a p-value of 1
        t_tests_2[f_ind] = stats.ttest_1samp(np.abs(np.array(success_rate_all_blocks_0) - np.array(success_rate_all_blocks_1)), 0)[1]
        # account for nan-values (e.g. when all differences are 0) -> assign high p-value
        if np.isnan(t_tests_2[f_ind]):
            t_tests_2[f_ind] = 1
        
    # Select next feature
    min_val_curr = min(t_tests_2)
    feat_min_p_val = [i for i in range(num_feat_not_sel) if t_tests_2[i] == min_val_curr]
    if len(feat_min_p_val) == 1:
        feat_sel.append(feat_not_sel[feat_min_p_val[0]])
        feat_sel_criteria.append("Min p-value: " + str(round(min_val_curr, 4)))
    # multiple best features -> choose one randomly
    else:
        feat_sel.append(feat_not_sel[random.choice(feat_min_p_val)])
        feat_sel_criteria.append("Min p-value: " + str(round(min_val_curr, 4)) + " random from " + str(feat_min_p_val))
        
    print("Feature selected:", feat_sel[-1])
    print("Criterion:", feat_sel_criteria[-1])

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