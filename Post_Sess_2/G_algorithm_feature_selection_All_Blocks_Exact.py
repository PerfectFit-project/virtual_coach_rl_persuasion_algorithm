'''
Final version of feature selection based on G-algorithm for experiment.
To be run after session 2
This time we compute one-sample t-tests across all blocks.
We also compute Q-values exactly based on approximated reward and transition functions.
The reward function depends also on the next state.
'''

import itertools
import math
import numpy as np
import pandas as pd
import pickle
import random
from scipy import stats

import Utils as util


def approx_dynamics_blocks(data, num_blocks_plus_1, num_act, blocks, 
                           blocks_plus_1, f, feat_sel):
    """Approximate reward and transition functions for blocks.

    Args:
        data (list): List with samples of the form <s0, s1, a, r>.
        num_blocks_plus_1 (int): Number of blocks when using 1 more than the so far selected features.
        num_act (int): Number of actions.
        blocks (list): Abstract states based on so far selected features.
        blocks_plus_1 (list): Abstract states based on selecting one more feature.
        f (int): Candidate feature
        feat_sel (list): Already selected features
        
    Returns:
        np-array: reward function
        np-array: transition function

    """
    
    num_samples = len(data)
    
    trans_func = np.zeros((num_blocks_plus_1, num_act, num_blocks_plus_1))
    reward_func = np.zeros((num_blocks_plus_1, num_act, num_blocks_plus_1))
    reward_func_count = np.zeros((num_blocks_plus_1, num_act, num_blocks_plus_1))

    for b_ind, block in enumerate(blocks):  # for each block based on so far selected features
    
        for data_index in range(num_samples):  # for each data sample
            
            s = data[data_index][0][f]  # value for current feature 
            s_b = np.take(data[data_index][0], feat_sel)  # values for already selected features
            
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
        for a in range(num_act):  # for each action
            summed = sum(trans_func[b_ind, a])
            if summed > 0:
                trans_func[b_ind, a] /= summed
            # If we have no data on a state-action combination, we assume equal probability of transitioning to each other state
            else:
                trans_func[b_ind, a] = np.ones(num_blocks_plus_1) / num_blocks_plus_1  # i.e. 0.5 probability of transitioning to each state
            for b_prime in range(num_blocks_plus_1):
                if reward_func_count[b_ind, a, b_prime] > 0:
                    reward_func[b_ind, a, b_prime] /= reward_func_count[b_ind, a, b_prime]
       
    return reward_func, trans_func


def approx_dynamics_single_feature(data, num_act, f):
    """Approximate reward and transition functions based on single features.

    Args:
        data (list): List with samples of the form <s0, s1, a, r>.
        num_act (int): Number of actions.
        f (int): feature
        
    Returns:
        np-array: reward function
        np-array: transition function

    """
    
    num_samples = len(data)
    
    trans_func = np.zeros((2, num_act, 2))
    reward_func = np.zeros((2, num_act, 2))
    reward_func_count = np.zeros((2, num_act, 2))
    for s in range(2):  # i.e. if feature is 0 and if feature is 1
        for data_index in range(num_samples):  # for each data sample
            if data[data_index][0][f] == s:
                a = data[data_index][2]
                s_prime = data[data_index][1][f]  # feature value of next state
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
                trans_func[s, a] = np.ones(2) / 2  # i.e. 0.5 probability of transitioning to each state
            for s_prime in range(2):
                if reward_func_count[s, a, s_prime] > 0:
                    reward_func[s, a, s_prime] /= reward_func_count[s, a, s_prime]
    
    return reward_func, trans_func


def feature_selection_level_3(data, effort_mean, feat_to_select, 
                              num_feat_to_select = 3, num_act = 5,
                              discount_factor = 0.85):
    """Select features for level 3 of algorithm complexity.

    Args:
        data (list): List with samples of the form <s0, s1, a, r>.
        effort_mean (float): Mean of effort responses.
        feat_to_select (list): Candidate features.
        num_feat_to_select (int): Number of features to select.
        num_act (int): Number of possible actions.
        discount_factor (float): Discount factor for Q-value computation.

    Returns:
        list: Selected features.
        list: Criteria for selected features.

    """

    # All effort responses
    list_of_efforts = list(np.array(data)[:, 3].astype(int))
    
    # Map effort responses to rewards from -1 to 1, with the mean mapped to 0.
    map_to_rewards = util.get_map_effort_reward(effort_mean, output_lower_bound = -1, 
                                                output_upper_bound = 1, 
                                                input_lower_bound = 0, 
                                                input_upper_bound = 10)
    reward_list = util.map_efforts_to_rewards(list_of_efforts, map_to_rewards)
    # now write these obtained reward values into "data" in place of the original rewards
    for i in range(len(reward_list)):
        data[i][3] = reward_list[i]

    num_feat = len(feat_to_select)
            
    # Select first feature
    q_values = np.zeros((num_feat, 2, num_act))
    t_tests = np.zeros((num_feat))
    
    for f in range(num_feat):
    
        # Compute transition function and reward function for feature
        reward_func, trans_func = approx_dynamics_single_feature(data, 
                                                                 num_act, f)
    
        # Value iteration for current feature    
        q_values[f], _ = util.get_Q_values_opt_policy(discount_factor, 
                                                      trans_func, 
                                                      reward_func, 
                                                      reward_dep_next_state = True)
        
        # t-test -> get p-value
        # one-sample t-test, comparing the absolute difference to 0
        t_tests[f] = stats.ttest_1samp(np.abs(np.array(q_values[f][0]) - np.array(q_values[f][1])), 0)[1]
    
    min_p_val = np.nanmin(t_tests)  # minimum p-value for t-test, ignoring nan-values
    feat_sel_options = [i for i in range(num_feat) if t_tests[i] == min_p_val]
    feat_sel = [random.choice(feat_sel_options)]  # choose randomly if there are multiple best features
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
        
        # all abstract states based on the so far selected features
        blocks = [list(i) for i in itertools.product([0, 1], repeat = num_feat_sel)]
        # all abstract states including the new feature
        blocks_plus_1 = [list(i) for i in itertools.product([0, 1], repeat = num_feat_sel + 1)]
        num_blocks_plus_1 = len(blocks_plus_1)
        
        q_values_2 = np.zeros((num_feat_not_sel, num_blocks_plus_1, num_act))
        t_tests_2 = np.zeros(num_feat_not_sel)
        
        for f_ind, f in enumerate(feat_not_sel):  # for each not yet selected feature
            
            # Approximate reward and transition function
            reward_func, trans_func = approx_dynamics_blocks(data, 
                                                             num_blocks_plus_1, 
                                                             num_act, blocks, 
                                                             blocks_plus_1, f,
                                                             feat_sel)
        
            # Now we need to compute the Q-values
            # Value iteration for current feature    
            q_values_2[f_ind], _ = util.get_Q_values_opt_policy(discount_factor, 
                                                                trans_func, 
                                                                reward_func, 
                                                                reward_dep_next_state = True)
            
            # Separate Q-values based on whether current candidate feature is 0 or 1
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
        
    return feat_sel, feat_sel_criteria
    

def compute_opt_policy_level_3(data, feat_sel, num_act = 5, 
                               discount_factor = 0.85):
    """Compute the optimal policy for level 3 of algorithm complexity.

    Args:
        data (list): List with samples of the form <s0, s1, a, r>.
        feat_sel (list): Features to consider for policy computation.
        num_act (int): Number of possible actions.
        discount_factor (float): Discount factor for Q-value computation.

    Returns:
        list: Optimal actions in each state.

    """
    
    num_samples = len(data)
    num_feat_selected = len(feat_sel)
    
    abstract_states = [list(i) for i in itertools.product([0, 1], repeat = num_feat_selected)]
    
    # Compute transition function and reward function
    # Reward function is also dependent on next state
    trans_func = np.zeros((int(2 ** num_feat_selected), num_act, int(2 ** num_feat_selected)))
    reward_func = np.zeros((int(2 ** num_feat_selected), num_act, int(2 ** num_feat_selected)))
    reward_func_count = np.zeros((int(2 ** num_feat_selected), num_act, int(2 ** num_feat_selected)))
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
                trans_func[s_ind, a] = np.ones(int(2 ** num_feat_selected)) / (2 ** num_feat_selected)
            for s_prime_ind in range(len(abstract_states)):
                if reward_func_count[s_ind, a, s_prime_ind] > 0:
                    reward_func[s_ind, a, s_prime_ind] /= reward_func_count[s_ind, a, s_prime_ind]
    
    # Value iteration        
    q_values_exact, _ = util.get_Q_values_opt_policy(discount_factor, trans_func, reward_func, 
                                                     reward_dep_next_state = True)
    opt_policy = [[[[a for a in range(num_act) if q_values_exact[abstract_states.index([i, j, k])][a] == max(q_values_exact[abstract_states.index([i, j, k])])] for k in range(2)] for j in range(2)] for i in range(2)]
    
    return opt_policy

    
if __name__ == "__main__":
    
    DISCOUNT_FACTOR = 0.85  # for computation of Q-values
    NUM_FEAT_TO_SELECT = 3  # number of features to select
    
    # Load data. Data has <s, s', a, r>-samples.
    feat_to_select = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ] # [0, 1, 2, 3, 4, 5, 6] in experiment for features [0, 1, 2, 3, 4, 6, 7]
    data  = pd.read_csv('W:/staff-umbrella/perfectfit/Exp0/Analysis/All_Data/rl_samples_list_binary_exp.csv', 
                        converters={'s0': eval, 's1': eval})
    data = data.values.tolist()
    
    # Mean value of effort responses
    with open("W:/staff-umbrella/perfectfit/Exp0/Analysis/All_Data/all_data_effort_mean", "rb") as f:
        effort_mean = pickle.load(f)
        
    # Select features
    feat_sel, feat_sel_criteria = feature_selection_level_3(data, effort_mean, 
                                                            feat_to_select,
                                                            num_act = 5, 
                                                            num_feat_to_select = NUM_FEAT_TO_SELECT,
                                                            discount_factor = DISCOUNT_FACTOR)
     
    # Store selected features
    with open('W:/staff-umbrella/perfectfit/Exp0/Analysis/All_Data/Level_3_G_algorithm_chosen_features', 'wb') as f:
        pickle.dump(feat_sel, f)
    with open("W:/staff-umbrella/perfectfit/Exp0/Analysis/All_Data/Level_3_G_algorithm_chosen_features_criteria", 'wb') as f:
        pickle.dump(feat_sel_criteria, f)
    
        
    # Compute optimal policy
    opt_policy = compute_opt_policy_level_3(data, feat_sel, num_act = 5, 
                                            discount_factor = DISCOUNT_FACTOR)
    
    with open('W:/staff-umbrella/perfectfit/Exp0/Analysis/All_Data/Level_3_Optimal_Policy', 'wb') as f:
        pickle.dump(opt_policy, f)
    