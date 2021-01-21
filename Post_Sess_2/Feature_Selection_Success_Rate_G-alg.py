'''
Feature selection as in G-algorithm but with success rates.
To be run after session 2.
'''
import numpy as np
import itertools
from scipy import stats
import pickle
import Utils as util
import pandas as pd

# load data. Data has <s, s', a, r> samples.
feat_to_select = [0, 1, 2, 3, 4, 6, 7]
data  = pd.read_csv('data_samples_post_sess_2.csv', converters={'s0': eval, 's1': eval})
data = data.values.tolist()

num_act = 4
num_feat = len(feat_to_select)
num_samples = len(data)

num_states = 2 ** num_feat
actions = [i for i in range(num_act)]

states = list(map(list, itertools.product([0, 1], repeat = num_feat)))

# Settings for calculation of Q-values
q_num_iter = 1000 * num_samples # num_samples = number of people
discount_factor = 0.85
alpha = 0.01

num_feat_to_select = 3 # number of features to select
        
# Select first feature
rewards = np.zeros((num_feat, 2, num_act))
trials = np.zeros((num_feat, 2, num_act))
t_tests = np.zeros((num_feat))
for f in range(num_feat):
   for data_index in range(num_samples): # for each data sample
        s = data[data_index][0][f]
        a = data[data_index][2]
        r = int(data[data_index][3])
             
        rewards[f][s][a] += r
        trials[f][s][a] += 1
    
   # t-test based on success-rates
   success_rate_0 = np.divide(rewards[f][0], trials[f][0], out=np.zeros_like(rewards[f][0]), where=trials[f][0]!=0)
   success_rate_1 = np.divide(rewards[f][1], trials[f][1], out=np.zeros_like(rewards[f][1]), where=trials[f][1]!=0)
   t_tests[f] = stats.ttest_ind(success_rate_0, success_rate_1)[0]

feat_sel = [np.argmin(abs(t_tests))]
feat_sel_criteria = ["First feature -> min. p-value."]
print("First feature selected:", feat_sel[0], "with p-value", np.min(abs(t_tests)))

# Select remaining features
for j in range(num_feat_to_select - 1):
    num_feat_sel = len(feat_sel)
    feat_not_sel = [f for f in range(num_feat) if not f in feat_sel]
    num_feat_not_sel = len(feat_not_sel)
    
    num_blocks = 2 ** num_feat_sel
    blocks = [list(i) for i in itertools.product([0, 1], repeat = num_feat_sel)]
    rewards_2 = np.zeros((num_blocks, num_feat_not_sel, 2, num_act))
    trials_2 = np.zeros((num_blocks, num_feat_not_sel, 2, num_act))
    t_tests_2 = np.zeros((num_blocks, num_feat_not_sel))
    
    # Compute t-test based on success-rates
    for b_ind, block in enumerate(blocks): # for each block
        for f_ind, f in enumerate(feat_not_sel): # for each not yet selected feature
            for data_index in range(num_samples): # for each data sample
                
                s = data[data_index][0][f]
                s_b = np.take(data[data_index][0], feat_sel)
                
                # both s must be in the current block
                if list(s_b) == block:
                    
                    a = data[data_index][2]
                    r = int(data[data_index][3])
                               
                    # save reward
                    rewards_2[b_ind][f_ind][s][a] += r
                    trials_2[b_ind][f_ind][s][a] += 1
            
            success_rate_1  = np.divide(rewards_2[b_ind][f_ind][0], trials_2[b_ind][f_ind][0],
                                        out = np.zeros_like(rewards_2[b_ind][f_ind][0]),
                                        where=trials_2[b_ind][f_ind][0]!=0)
            success_rate_1  = np.divide(rewards_2[b_ind][f_ind][1], trials_2[b_ind][f_ind][1],
                                        out = np.zeros_like(rewards_2[b_ind][f_ind][1]),
                                        where=trials_2[b_ind][f_ind][1]!=0)
            # t-test
            t_tests_2[b_ind, f_ind] = stats.ttest_ind(success_rate_0, success_rate_1)[0]
    
    # Select next feature
    feat_sel, feat_sel_criteria = util.feat_sel_num_blocks_avg_p_val(feat_not_sel, num_feat_not_sel, 
                                                                     blocks, 
                                                                     t_tests_2, feat_sel,
                                                                     feat_sel_criteria)
        
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

optimal_policy = [[[[a for a in range(num_act) if success_rates[i, j, k, a] == max(success_rates[i, j, k])] for k in range(2)] for j in range(2)] for i in range(2)]

with open('Level_2_Optimal_Policy', 'wb') as f:
    pickle.dump(optimal_policy, f)