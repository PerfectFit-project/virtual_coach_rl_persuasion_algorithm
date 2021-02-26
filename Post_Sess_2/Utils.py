'''
Common functions
'''
import numpy as np
import sqlite3
import random
from copy import deepcopy

ATTENTION_CHECK_1_TRUE = 4
ATTENTION_CHECK_2_TRUE = 0

def feat_sel_num_blocks_avg_p_val(feat_not_sel, num_feat_not_sel, blocks, 
                                  t_tests_2, feat_sel, feat_sel_criteria):
    """
    Best feature first based on num. of blocks in which it is best, then based on avg. p-value for best blocks, then randomly."
    """
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

def gather_data_post_sess_2(database_path, feat_to_select = [0, 1, 2, 3, 4, 6, 7]):
    """
    Gathers data from sqlite database after session 2.
    
    Args:
        database_path: path to database
        feat_to_select: which of the state features to consider
    """
    num_select = len(feat_to_select)
    
    # create db connection
    try:
        sqlite_connection = sqlite3.connect(database_path)
        cursor = sqlite_connection.cursor()
        print("Connection created")
        sqlite_select_query = """SELECT * from users WHERE state_0 IS NOT NULL"""
        cursor.execute(sqlite_select_query)
        data_db = cursor.fetchall()
        cursor.close()

    except sqlite3.Error as error:
        print("Error while connecting to sqlite", error)
    finally:
        if (sqlite_connection):
            sqlite_connection.close()
            print("Connection closed")
 
    num_rows = len(data_db)
    all_states = [] # to compute mean values for the features
    
    data = [] # to save transitions
    user_ids = []
    
    # for each user that has completed at least 1 session
    for row in range(num_rows):
        
        # Get first state
        s0_arr = np.array([int(i) for i in data_db[row][20].split('|')])[feat_to_select]
        s0 = list(s0_arr)
        
        # Attention check question answers for first session
        check_state_1 = [int(i) for i in data_db[row][16].split('|')][0]
        check_state_2 = [int(i) for i in data_db[row][17].split('|')][0]
        
        # Whether the first session has passed enough attention checks
        passed_check_state = pass_attention_checks(check_state_1, check_state_2)
        
        # needed to later compute the mean values per feature
        if passed_check_state:
            all_states.append(s0_arr)
        
        # Check if the user has completed a second session
        if not data_db[row][21] == None:
            
            s1_arr = np.array([int(i) for i in data_db[row][21].split('|')])[feat_to_select]
            s1 = list(s1_arr)
        
            # attention check answers for second session
            check_next_1 = [int(i) for i in data_db[row][16].split('|')][1]
            check_next_2 = [int(i) for i in data_db[row][17].split('|')][1]
        
            # Whether the second session has passed enough attention checks
            passed_check_next_state = pass_attention_checks(check_next_1, check_next_2)
        
            # need to pass at least 1 out of 2 attention checks in each of the two sessions
            passed_attention_checks = passed_check_state and passed_check_next_state
            
            # the transition is not used if the attention check criteria are not met
            if passed_attention_checks:
            
                a = [int(i) for i in data_db[row][25].split('|')][0]
                r = [int(i) for i in data_db[row][7].split('|')][0]
            
                data.append([s0, s1, a, r]) # save transition
                user_ids.append(data_db[row][0]) # save corresponding user ID
              
            # Even if the transition is not used, a state may still be used to compute 
            # mean values for the features
            if passed_check_next_state:
                all_states.append(s1_arr)
    
    all_states = np.array(all_states)
        
    # compute the mean value for each feature
    feat_means = np.mean(all_states, axis = 0)
    
    # convert features to binary features based on mean values
    for row in range(len(data)):
        data[row][0] = [1 if data[row][0][i] >= feat_means[i] else 0 for i in range(num_select)]
        data[row][1] = [1 if data[row][1][i] >= feat_means[i] else 0 for i in range(num_select)]
    
    return data, feat_means, user_ids

def check_attention_checks_session(database_path, session_num):
    """
    Returns IDs of users who have passed/failed a session based on 
    attention checks.
    
    Args:
        database_path: path to database
        session_num: which session to check for; from 1 to 5
    """
    # create db connection
    try:
        sqlite_connection = sqlite3.connect(database_path)
        cursor = sqlite_connection.cursor()
        print("Connection created")
        if session_num == 1:
            sqlite_select_query = """SELECT * from users WHERE state_0 IS NOT NULL"""
        elif session_num == 2:
            sqlite_select_query = """SELECT * from users WHERE state_1 IS NOT NULL"""
        elif session_num == 3:
            sqlite_select_query = """SELECT * from users WHERE state_2 IS NOT NULL"""
        elif session_num == 4:
            sqlite_select_query = """SELECT * from users WHERE state_3 IS NOT NULL"""
        elif session_num == 5:
            sqlite_select_query = """SELECT * from users WHERE state_4 IS NOT NULL"""
        cursor.execute(sqlite_select_query)
        data_db = cursor.fetchall()
        cursor.close()

    except sqlite3.Error as error:
        print("Error while connecting to sqlite", error)
    finally:
        if (sqlite_connection):
            sqlite_connection.close()
            print("Connection closed")
 
    num_rows = len(data_db)
    user_ids_passed = []
    user_ids_failed = []
    
    session_index = session_num - 1
    
    # for each user that has completed at least 1 session
    for row in range(num_rows):
        
        # Attention check question answers
        check_state_1 = [int(i) for i in data_db[row][16].split('|')][session_index]
        check_state_2 = [int(i) for i in data_db[row][17].split('|')][session_index]
        
        # Whether the first session has passed enough attention checks
        passed_check = pass_attention_checks(check_state_1, check_state_2)
       
        # save corresponding user ID
        if passed_check:
            user_ids_passed.append(data_db[row][0]) 
        else:
            user_ids_failed.append(data_db[row][0]) 
    
    return user_ids_passed, user_ids_failed

def pass_attention_checks(answer1, answer2):
    """
    Returns whether at least 1/2 attention checks were passed for a session.
    
    Args:
        answer1: given answer for attention check 1
        answer2: given answer for attention check 2
    """
    return answer1 == ATTENTION_CHECK_1_TRUE or answer2 == ATTENTION_CHECK_2_TRUE

def gather_data_post_sess_5(database_path, feat_to_select = [0, 1, 2, 3, 4, 6, 7]):
    """
    Gathers data from sqlite database after session 5.
    Also considers whether 2/2 attention checks have been failed in a session
    and removes such samples.
    
    Args:
        database_path: path to database
        feat_to_select: which of the state features to consider
    """
    num_select = len(feat_to_select)
    
    # create db connection
    try:
        sqlite_connection = sqlite3.connect(database_path)
        cursor = sqlite_connection.cursor()
        print("Connection created")
        sqlite_select_query = """SELECT * from users WHERE state_0 IS NOT NULL"""
        cursor.execute(sqlite_select_query)
        data_db = cursor.fetchall()
        cursor.close()

    except sqlite3.Error as error:
        print("Error while connecting to sqlite", error)
    finally:
        if (sqlite_connection):
            sqlite_connection.close()
            print("Connection closed")
 
    num_rows = len(data_db)
    all_states = []
    
    data = []
    user_ids = [] # user ID for each sample
    
    for row in range(num_rows): # for each person with at least 1 session
    
        user_id_curr = data_db[row][0] # ID of current person
        
        for state_ind, state in enumerate([20, 21, 22, 23]): # for session 1-4
        
            if not data_db[row][state] == None: # ensure current state is not None
            
                s0_arr = np.array([int(i) for i in data_db[row][state].split('|')])[feat_to_select]
                s0 = list(s0_arr)
            
                # Attention check question answers for first state of transition
                check_state_1 = [int(i) for i in data_db[row][16].split('|')][state_ind]
                check_state_2 = [int(i) for i in data_db[row][17].split('|')][state_ind]
                passed_check_state = pass_attention_checks(check_state_1, check_state_2)
                
                # needed to later compute the mean values per feature
                # Make sure to add each state only once
                if state_ind == 0 and passed_check_state:
                    all_states.append(s0_arr)
            
                if not data_db[row][state + 1] == None: # ensure next state is not None
                
                    s1_arr = np.array([int(i) for i in data_db[row][state + 1].split('|')])[feat_to_select]
                    s1 = list(s1_arr)
                
                    # Attention check question answers for second state of transition
                    check_next_1 = [int(i) for i in data_db[row][16].split('|')][state_ind + 1]
                    check_next_2 = [int(i) for i in data_db[row][17].split('|')][state_ind + 1]
                    passed_check_next_state = pass_attention_checks(check_next_1, check_next_2)
                    
                    # need to pass at least 1 out of 2 attention checks in each of the two sessions
                    passed_attention_checks = passed_check_state and passed_check_next_state
                    
                    # the transition is not used if the attention check criteria are not met
                    if passed_attention_checks:
                        
                        # get action and reward
                        a = [int(i) for i in data_db[row][25].split('|')][state_ind]
                        r = [int(i) for i in data_db[row][7].split('|')][state_ind]
                    
                        data.append([s0, s1, a, r]) # save transition
                        user_ids.append(user_id_curr) # save corresponding user ID
                        
                    # needed to later compute the mean values per feature
                    if passed_check_next_state:
                        all_states.append(s1_arr)
                        
    all_states = np.array(all_states)
        
    # compute the mean value for each feature
    feat_means = np.mean(all_states, axis = 0)
    
    # convert features to binary features based on mean values
    for row in range(len(data)):
        data[row][0] = [1 if data[row][0][i] >= feat_means[i] else 0 for i in range(num_select)]
        data[row][1] = [1 if data[row][1][i] >= feat_means[i] else 0 for i in range(num_select)]
    
    return data, feat_means, user_ids

def policy_evaluation(observation_space_size, discount_factor, trans_func,
                      reward_func, policy, q_vals, 
                      update_tolerance = 0.000000001):
    """
    Returns the q-values for a specific policy.
    
    Args:
        observation_space_size: number of observations
        discount_factor: discount factor of MDP
        trans_func: transition function
        reward_func: reward function
        policy: policy to compute q-values for
        update_tolerance: precision
    """
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
    """
    Returns the q-values for each state under the optimal policy.
    
    Args:
        discount_factor: discount factor of MDP
        trans_func: transition function (dim.: num_states x num_actions x num_states)
        reward_func: reward function (dim.: num_states x num_actions)
    """
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