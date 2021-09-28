'''
Final version of feature selection based on G-algorithm for experiment.
To be run after session 2
This time we compute one-sample t-tests across all blocks.
We also compute Q-values exactly based on approximated reward and transition functions.
The reward function depends also on the next state.
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
data  = pd.read_csv('W:/staff-umbrella/perfectfit/Exp0/Analysis/All_Data/rl_samples_list_binary_exp.csv', converters={'s0': eval, 's1': eval})
data = data.values.tolist()

# All effort responses
list_of_efforts = list(np.array(data)[:, 3].astype(int))
# Mean value of effort responses
with open("W:/staff-umbrella/perfectfit/Exp0/Analysis/All_Data/all_data_effort_mean", "rb") as f:
    effort_mean = pickle.load(f)
# Map effort responses to rewards from -1 to 1, with the mean mapped to 0.
map_to_rewards = util.get_map_effort_reward(effort_mean, output_lower_bound = -1, 
                                            output_upper_bound = 1, 
                                            input_lower_bound = 0, 
                                            input_upper_bound = 10)
reward_list = util.map_efforts_to_rewards(list_of_efforts, map_to_rewards)
# now write these obtained reward values into "data" in place of the original rewards
for i in range(len(reward_list)):
    data[i][3] = reward_list[i]

num_act = 5 # number of actions
num_feat = len(feat_to_select)
num_samples = len(data)

num_states = 2 ** num_feat
actions = [i for i in range(num_act)]

states = list(map(list, itertools.product([0, 1], repeat = num_feat)))

# Settings for calculation of Q-values
discount_factor = 0.85

num_feat_to_select = 3 # number of features to select
        
# Select first feature
q_values = np.zeros((num_feat, 2, num_act))
t_tests = np.zeros((num_feat))

for f in range(num_feat):

    # Compute transition function and reward function
    trans_func = np.zeros((2, num_act, 2))
    reward_func = np.zeros((2, num_act, 2))
    reward_func_count = np.zeros((2, num_act, 2))
    for s in range(2): # i.e. if feature is 0 and if feature is 1
        for data_index in range(num_samples): # for each data sample
            if data[data_index][0][f] == s:
                a = data[data_index][2]
                s_prime = data[data_index][1][f] # feature value of next state
                r = data[data_index][3]
                
                # update statistics
                trans_func[s, a, s_prime] += 1
                reward_func[s, a, s_prime] += r
                reward_func_count[s, a, s_prime] += 1
   
        # Normalize transition and reward function for current value of feature
        for a in range(num_act):
            summed = sum(trans_func[s, a])
            if summed > 0:
                trans_func[s, a] /= summed
            # if we have no data on a state-action combination, we assume equal probability of transitioning to each other state
            else:
                trans_func[s, a] = np.ones(2) / 2 # i.e. 0.5 probability of transitioning to each state
            for s_prime in range(2):
                if reward_func_count[s, a, s_prime] > 0:
                    reward_func[s, a, s_prime] /= reward_func_count[s, a, s_prime]

    # Value iteration for current feature    
    q_values[f], _ = util.get_Q_values_opt_policy(discount_factor, trans_func, reward_func, 
                                                  reward_dep_next_state = True)
    
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
    
    # number of abstract states based on the so far selected features
    num_blocks = 2 ** num_feat_sel
    # all abstract states based on the so far selected features
    blocks = [list(i) for i in itertools.product([0, 1], repeat = num_feat_sel)]
    # all abstract states including the new feature
    blocks_plus_1 = [list(i) for i in itertools.product([0, 1], repeat = num_feat_sel + 1)]
    num_blocks_plus_1 = len(blocks_plus_1)
    
    q_values_2 = np.zeros((num_feat_not_sel, num_blocks_plus_1, num_act))
    t_tests_2 = np.zeros(num_feat_not_sel)
    
    for f_ind, f in enumerate(feat_not_sel): # for each not yet selected feature
    
        trans_func = np.zeros((num_blocks_plus_1, num_act, num_blocks_plus_1))
        reward_func = np.zeros((num_blocks_plus_1, num_act, num_blocks_plus_1))
        reward_func_count = np.zeros((num_blocks_plus_1, num_act, num_blocks_plus_1))
    
        for b_ind, block in enumerate(blocks): # for each block based on so far selected features
        
            for data_index in range(num_samples): # for each data sample
                
                s = data[data_index][0][f] # value for current feature 
                s_b = np.take(data[data_index][0], feat_sel) # values for already selected features
                
                # the start state must be in the current block
                if list(s_b) == block:
                    
                    # next state and next state block
                    s_prime = data[data_index][1][f]
                    s_prime_b = np.take(data[data_index][1], feat_sel)
                    # get the action and reward
                    a = data[data_index][2]
                    r = data[data_index][3]
                    
                    # update statistics
                    trans_func[blocks_plus_1.index(list(s_b) + [s]), a, blocks_plus_1.index(list(s_prime_b) + [s_prime])] += 1
                    reward_func[blocks_plus_1.index(list(s_b) + [s]), a, blocks_plus_1.index(list(s_prime_b) + [s_prime])] += r
                    reward_func_count[blocks_plus_1.index(list(s_b) + [s]), a, blocks_plus_1.index(list(s_prime_b) + [s_prime])] += 1
        
        # Now need to normalize the reward and transition functions
        # For each block including the current candidate feature
        for b_ind in range(num_blocks_plus_1):  
            for a in range(num_act): # for each action
                summed = sum(trans_func[b_ind, a])
                if summed > 0:
                    trans_func[b_ind, a] /= summed
                # if we have no data on a state-action combination, we assume equal probability of transitioning to each other state
                else:
                    trans_func[b_ind, a] = np.ones(num_blocks_plus_1) / num_blocks_plus_1 # i.e. 0.5 probability of transitioning to each state
                for b_prime in range(num_blocks_plus_1):
                    if reward_func_count[b_ind, a, b_prime] > 0:
                        reward_func[b_ind, a, b_prime] /= reward_func_count[b_ind, a, b_prime]
           
        # Now we need to compute the Q-values
        # Value iteration for current feature    
        q_values_2[f_ind], _ = util.get_Q_values_opt_policy(discount_factor, trans_func, reward_func, 
                                                            reward_dep_next_state = True)
        
        # separate Q-values based on whether current candidate feature is 0 or 1
        q_values_all_blocks_0 = np.array([q_values_2[f_ind][b_ind_even] for b_ind_even in range(0, len(blocks_plus_1), 2)]).flatten()
        q_values_all_blocks_1 = np.array([q_values_2[f_ind][b_ind_odd] for b_ind_odd in range(1, len(blocks_plus_1), 2)]).flatten()
        
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
with open('W:/staff-umbrella/perfectfit/Exp0/Analysis/All_Data/Level_3_G_algorithm_chosen_features', 'wb') as f:
    pickle.dump(feat_sel, f)
with open("W:/staff-umbrella/perfectfit/Exp0/Analysis/All_Data/Level_3_G_algorithm_chosen_features_criteria", 'wb') as f:
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
# Reward function is also dependent on next state
trans_func = np.zeros((int(2 ** num_feat_to_select), num_act, int(2 ** num_feat_to_select)))
reward_func = np.zeros((int(2 ** num_feat_to_select), num_act, int(2 ** num_feat_to_select)))
reward_func_count = np.zeros((int(2 ** num_feat_to_select), num_act, int(2 ** num_feat_to_select)))
for s_ind, s in enumerate(abstract_states):
    for data_index in range(num_samples):
        if list(np.take(np.array(data[data_index][0]), feat_sel)) == s:
            # index of next state
            next_state_index = abstract_states.index(list(np.take(data[data_index][1], feat_sel)))
            
            trans_func[s_ind, data[data_index][2], next_state_index] += 1
            r = data[data_index][3]
            reward_func[s_ind, data[data_index][2], next_state_index] += r
            reward_func_count[s_ind, data[data_index][2], next_state_index] += 1
   
    # Normalize reward and transition function
    for a in range(num_act):
        summed = sum(trans_func[s_ind, a])
        if summed > 0:
            trans_func[s_ind, a] /= summed
        # if we have no data on a state-action combination, we assume equal probability of transitioning to each other state
        else:
            trans_func[s_ind, a] = np.ones(int(2 ** num_feat_to_select)) / (2 ** num_feat_to_select)
        for s_prime_ind in range(len(abstract_states)):
            if reward_func_count[s_ind, a, s_prime_ind] > 0:
                reward_func[s_ind, a, s_prime_ind] /= reward_func_count[s_ind, a, s_prime_ind]

# Value iteration        
q_values_exact, _ = util.get_Q_values_opt_policy(discount_factor, trans_func, reward_func, 
                                                 reward_dep_next_state = True)
opt_policy = [[[[a for a in range(num_act) if q_values_exact[abstract_states.index([i, j, k])][a] == max(q_values_exact[abstract_states.index([i, j, k])])] for k in range(2)] for j in range(2)] for i in range(2)]

with open('W:/staff-umbrella/perfectfit/Exp0/Analysis/All_Data/Level_3_Optimal_Policy', 'wb') as f:
    pickle.dump(opt_policy, f)