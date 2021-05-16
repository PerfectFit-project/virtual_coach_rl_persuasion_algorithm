'''
Final version of feature selection based on G-algorithm for experiment.
To be run after session 2
This time we compute one-sample t-tests across all blocks.
'''
import numpy as np
import itertools
from scipy import stats
import pickle
import Utils as util
import pandas as pd
import math
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
# Map effort responses to rewards from -1 to 1, with the mean mapped to 1.
map_to_rewards = util.get_map_effort_reward(effort_mean, output_lower_bound = -1, 
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

# Settings for calculation of Q-values
q_num_iter = 50000 * num_samples # num_samples = num people after session 2
q_num_iter_final = 100000 * num_samples # num_samples = num people after session 2
discount_factor = 0.85
alpha = 0.01

num_feat_to_select = 3 # number of features to select
        
# Select first feature
q_values = np.zeros((num_feat, 2, num_act))
t_tests = np.zeros((num_feat))
for f in range(num_feat):
    
    # Compute Q-values
    for t in range(q_num_iter):
        data_index = np.random.randint(0, num_samples)
        s = data[data_index][0][f]
        s_prime = data[data_index][1][f]
        a = data[data_index][2]
        r = data[data_index][3]
                   
        # TD Update 
        best_next_action = np.argmax(q_values[f, s_prime])     
        td_target = r + discount_factor * q_values[f, s_prime][best_next_action] 
        td_delta = td_target - q_values[f][s][a] 
        q_values[f][s][a] += alpha * td_delta 
    
    # t-test -> get p-value
    # one-sample t-test, comparing the absolute difference to 0
    t_tests[f] = stats.ttest_1samp(np.abs(np.array(q_values[f][0]) - np.array(q_values[f][1])), 0)[1]

min_p_val = np.nanmin(t_tests) # minimum p-value for t-test, ignoring nan-values
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
    q_values_2 = np.zeros((num_blocks, num_feat_not_sel, 2, num_act))
    t_tests_2 = np.zeros(num_feat_not_sel)
    
    for f_ind, f in enumerate(feat_not_sel): # for each not yet selected feature
    
        q_values_all_blocks_0 = []
        q_values_all_blocks_1 = []
    
        for b_ind, block in enumerate(blocks): # for each block
            for t in range(q_num_iter):
                
                data_index = np.random.randint(0, num_samples) # data sample
                s = data[data_index][0][f]
                s_b = np.take(data[data_index][0], feat_sel)
                s_prime = data[data_index][1][f]
                s_prime_b = np.take(data[data_index][1], feat_sel)
                
                # both s and s' must be in the current block
                if list(s_b) == block and list(s_prime_b) == block:
                    
                    # get the action and reward
                    a = data[data_index][2]
                    r = data[data_index][3]
                               
                    # TD Update 
                    best_next_action = np.argmax(q_values_2[b_ind, f_ind, s_prime])     
                    td_target = r + discount_factor * q_values_2[b_ind, f_ind, s_prime, best_next_action] 
                    td_delta = td_target - q_values_2[b_ind, f_ind, s, a] 
                    q_values_2[b_ind][f_ind][s][a] += alpha * td_delta 
            
            # append data from this block to data from previous blocks
            q_values_all_blocks_0 += list(q_values_2[b_ind][f_ind][0])
            q_values_all_blocks_1 += list(q_values_2[b_ind][f_ind][1])
        
        # Now that we have the data on all blocks, we run a one-sample t-test
        t_value = stats.ttest_1samp(np.abs(np.array(q_values_all_blocks_0) - np.array(q_values_all_blocks_1)), 0)[1]
        # If all differences are 0, the p-value will be nan -> so set the p-value high, i.e. 1
        if math.isnan(t_value):
            t_value = 1
        t_tests_2[f_ind] = t_value
    
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
with open('Level_3_G_algorithm_chosen_features', 'wb') as f:
    pickle.dump(feat_sel, f)
with open("Level_3_G_algorithm_chosen_features_criteria", 'wb') as f:
    pickle.dump(feat_sel_criteria, f)

'''    
# Compute Q-values
q_values = np.zeros((2, 2, 2, num_act))
for t in range(q_num_iter_final):
    data_index = np.random.randint(0, num_samples)
    s = np.take(np.array(data[data_index][0]), feat_sel)
    s_prime = np.take(np.array(data[data_index][1]), feat_sel)
    a = data[data_index][2]
    r = int(data[data_index][3])
    if r == 0:
        r = -1
               
    # TD Update 
    best_next_action = np.argmax(q_values[s_prime[0], s_prime[1], s_prime[2]])     
    td_target = r + discount_factor * q_values[s_prime[0], s_prime[1], s_prime[2], best_next_action] 
    td_delta = td_target - q_values[s[0], s[1], s[2], a] 
    q_values[s[0], s[1], s[2], a]  += alpha * td_delta 
    
opt_policy = [[[[a for a in range(num_act) if q_values[i, j, k, a] == max(q_values[i, j, k])] for k in range(2)] for j in range(2)] for i in range(2)]
'''

abstract_states = [list(i) for i in itertools.product([0, 1], repeat = num_feat_to_select)]

# Compute transition function and reward function
trans_func = np.zeros((int(2 ** num_feat_to_select), num_act, int(2 ** num_feat_to_select)))
reward_func = np.zeros((int(2 ** num_feat_to_select), num_act))
reward_func_count = np.zeros((int(2 ** num_feat_to_select), num_act))
for s_ind, s in enumerate(abstract_states):
    for data_index in range(num_samples):
        if list(np.take(np.array(data[data_index][0]), feat_sel)) == s:
            trans_func[s_ind, data[data_index][2], abstract_states.index(list(np.take(data[data_index][1], feat_sel)))] += 1
            r = data[data_index][3]
            reward_func[s_ind, data[data_index][2]] += r
            reward_func_count[s_ind, data[data_index][2]] += 1
   
    # Normalize
    for a in range(num_act):
        summed = sum(trans_func[s_ind, a])
        if summed > 0:
            trans_func[s_ind, a] /= summed
        # if we have no data on a state-action combination, we assume equal probability of transitioning to each other state
        else:
            trans_func[s_ind, a] = np.ones(int(2 ** num_feat_to_select)) / (2 ** num_feat_to_select)
        if reward_func_count[s_ind, a] > 0:
            reward_func[s_ind, a] /= reward_func_count[s_ind, a]

# Value iteration        
q_values_exact, _ = util.get_Q_values_opt_policy(discount_factor, trans_func, reward_func)
opt_policy = [[[[a for a in range(num_act) if q_values_exact[abstract_states.index([i, j, k])][a] == max(q_values_exact[abstract_states.index([i, j, k])])] for k in range(2)] for j in range(2)] for i in range(2)]

with open('Level_3_Optimal_Policy', 'wb') as f:
    pickle.dump(opt_policy, f)