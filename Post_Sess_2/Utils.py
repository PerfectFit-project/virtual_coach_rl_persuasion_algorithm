'''
Common functions
'''
import numpy as np
import sqlite3
import random
from copy import deepcopy

def feat_sel_num_blocks_avg_p_val(feat_not_sel, num_feat_not_sel, blocks, 
                                  t_tests_2, feat_sel, feat_sel_criteria):
    '''
    Best feature first based on num. of blocks in which it is best, then based on avg. p-value for best blocks, then randomly."
    '''
    num_best_per_feat = np.zeros(num_feat_not_sel) # num. of blocks in which each feat is best
    p_val_per_feat = np.zeros(num_feat_not_sel)
    for b_ind, block in enumerate(blocks):
        min_value = min(abs(t_tests_2[b_ind]))
        best_indices = [i for i in range(len(t_tests_2[b_ind])) if abs(t_tests_2[b_ind][i]) == min_value]
        num_best_indices = len(best_indices)
        for best_index in best_indices:
            num_best_per_feat[best_index] += 1 / num_best_indices
            p_val_per_feat[best_index] += min_value / num_best_indices
            print("Best feature if", feat_sel, "=", block, ":", feat_not_sel[best_index], "with p-value", np.round(min_value, 4))
    
    max_num_blocks = max(num_best_per_feat)
    criterion = "Max. num blocks with lowest p-value -> " + str(round(max_num_blocks, 4)) + " blocks"
    best_feature_indices = [i for i in range(num_feat_not_sel) if num_best_per_feat[i] == max_num_blocks]
    # Get the corresponding features
    best_feature_indices_features = [feat_not_sel[i] for i in best_feature_indices]
    if len(best_feature_indices) == 1:
        feat_sel.append(best_feature_indices_features[0])
        
    else:
        best_features_avg_p_vals = [p_val_per_feat[i]/num_best_per_feat[i] for i in best_feature_indices]
        min_avg_p_val = min(best_features_avg_p_vals)
        feat_lowest_avg_p_val_list = [i for i in range(len(best_features_avg_p_vals)) if best_features_avg_p_vals[i] == min_avg_p_val]
        # choose randomly if there are multiple features with lowest avg p-val
        feat_lowest_avg_p_val = random.choice(feat_lowest_avg_p_val_list) 
        feat_sel.append(feat_not_sel[best_feature_indices[feat_lowest_avg_p_val]])
        criterion += ", lowest avg. p-value for best blocks among " + str(best_feature_indices_features) + " with avg. " + str(min_avg_p_val)
        if len(feat_lowest_avg_p_val_list) > 1:
            criterion += " random from " + str([feat_not_sel[best_feature_indices[i]] for i in feat_lowest_avg_p_val_list])
    
    feat_sel_criteria.append(criterion)
    
    return feat_sel, feat_sel_criteria

def gather_data_post_sess_2(feat_to_select = [0, 1, 2, 3, 4, 6, 7]):
    '''
    Gathers data from sqlite database after session 2.
    feat_to_select: which of the state features to consider
    '''
    num_select = len(feat_to_select)
    
    # create db connection
    try:
        #sqlite_connection = sqlite3.connect('chatbot.db')
        sqlite_connection = sqlite3.connect('c:/users/nele2/CA/db_scripts/chatbot.db')
        cursor = sqlite_connection.cursor()
        print("Connection created")
        sqlite_select_query = """SELECT * from users WHERE state_1 IS NOT NULL"""
        cursor.execute(sqlite_select_query)
        data_db = cursor.fetchall()
        cursor.close()

    except sqlite3.Error as error:
        print("Error while connecting to sqlite", error)
    finally:
        if (sqlite_connection):
            sqlite_connection.close()
            print("Connection closed")
 
    #print(data_db)
    num_rows = len(data_db)
    all_states = np.zeros((num_rows * 2, num_select))
    
    data = []
    user_ids = []
    
    for row in range(num_rows):
        s0_arr = np.array([int(i) for i in data_db[row][20].split('|')])[feat_to_select]
        s0 = list(s0_arr)
        s1_arr = np.array([int(i) for i in data_db[row][21].split('|')])[feat_to_select]
        s1 = list(s1_arr)
        a = [int(i) for i in data_db[row][25].split('|')][0]
        r = [int(i) for i in data_db[row][7].split('|')][0]
        
        data.append([s0, s1, a, r])
        user_ids.append(data_db[row][0])
        
        # needed to later compute the mean values per feature
        all_states[row * 2] = s0_arr
        all_states[row * 2 + 1] = s1_arr
        
    # compute the mean value for each feature
    feat_means = np.mean(all_states, axis = 0)
    
    # convert features to binary features based on mean values
    for row in range(num_rows):
        data[row][0] = [1 if data[row][0][i] >= feat_means[i] else 0 for i in range(num_select)]
        data[row][1] = [1 if data[row][1][i] >= feat_means[i] else 0 for i in range(num_select)]
    
    return data, feat_means, user_ids

def policy_evaluation(observation_space_size, discount_factor, trans_func,
                      reward_func, policy, q_vals, 
                      update_tolerance = 0.000000001):
    '''
    Returns the q-values for a specific policy.
    
    observation_space_size: number of observations
    discount_factor: discount factor of MDP
    trans_func: transition function
    reward_func: reward function
    policy: policy to compute q-values for
    update_tolerance: precision
    '''
    q_vals_new = deepcopy(q_vals)
    num_act = len(trans_func[0])
   
    update = 1
    while update > update_tolerance:
        update = 0
        for s in range(observation_space_size):
            for a in range(num_act):
                q_vals_new[s, a] = reward_func[s, a] + discount_factor * sum([trans_func[s, a, s_prime] * q_vals[s_prime, int(policy[s_prime])] for s_prime in range(observation_space_size)])
                update = max(update, abs(q_vals[s, a] - q_vals_new[s, a]))
        q_vals = deepcopy(q_vals_new)
        
    return q_vals_new

def get_Q_values_opt_policy(discount_factor, trans_func, reward_func):
    '''
    Returns the q-values for each state under the optimal policy.
    
    discount_factor: discount factor of MDP
    trans_func: transition function (dim.: num_states x num_actions x num_states)
    reward_func: reward function (dim.: num_states x num_actions)
    '''
    min_iterations = 10 
    num_states = len(trans_func)
    num_act = len(trans_func[0])
    q_vals = np.zeros((num_states, num_act))

    
    policy = np.zeros(num_states)
    policy_new = np.ones(num_states)
    it = 0
    
    while not np.array_equal(policy, policy_new) or it < min_iterations:
        q_vals = policy_evaluation(num_states, discount_factor, 
                                   trans_func, reward_func, policy, q_vals)
        policy = policy_new
        policy_new = np.array([np.argmax(q_vals[s]) for s in range(num_states)])
        it += 1
    
    return q_vals, policy_new