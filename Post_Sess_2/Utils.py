'''
Common functions.
'''
from copy import deepcopy
import numpy as np
import sqlite3


ATTENTION_CHECK_1_TRUE = [3, 4] # "agree" or "agree strongly"
ATTENTION_CHECK_2_TRUE = [0, 1] # "disagree" or "disagree strongly"

AW_LIKERT_SCALE = ['0', '1', '2', '3', '4'] # answer options for questions with 5-point Likert scale


def get_map_effort_reward(effort_mean, output_lower_bound, 
                          output_upper_bound = 1, input_lower_bound = 0, 
                          input_upper_bound = 10):
    """
    Computes a mapping from effort responses to rewards.
    
    Args:
        effort_mean (float): mean effort response, to be mapped to halfway 
                             between output_lower_bound and output_upper_bound
        output_lower_bound (float): lowest value on output scale
        output_upper_bound (float, default 1): highest value on output scale
        input_lower_bound (int, default 0): lowest value on input scale, to 
                                            be mapped to output_lower_bound
        input_upper_bound (int, default 10): highest value on input scale, 
                                             to be mapped to output_upper_bound
                                             
    Returns:
        dictionary: maps effort responses (integers) to output scale values (float)
    """
    map_to_rew = {}
    
    # We can already map the endpoints of the input scale to the output scale
    map_to_rew[input_lower_bound] = output_lower_bound
    map_to_rew[input_upper_bound] = output_upper_bound
    
    # The mean value on the output scale
    mean_output = (output_upper_bound - output_lower_bound)/2 + output_lower_bound
    output_length_half = mean_output - output_lower_bound
    
    input_length_lower_half = effort_mean - input_lower_bound
    input_length_upper_half = input_upper_bound - effort_mean
    
    inc_lower_half = output_length_half / input_length_lower_half
    inc_upper_half = output_length_half / input_length_upper_half
    
    # Compute output scale values for input scale below mean.
    idx = 1
    for i in range(input_lower_bound + 1, int(np.ceil(effort_mean))):
        map_to_rew[i] = output_lower_bound + idx * inc_lower_half
        idx += 1
    
    # Compute output scale values for input scale above mean.
    idx = 1
    for i in range(input_upper_bound - 1, int(np.floor(effort_mean)), -1):
        map_to_rew[i] = output_upper_bound - idx * inc_upper_half
        idx += 1
    
    if np.floor(effort_mean) == effort_mean:
        map_to_rew[effort_mean] = mean_output
    
    # Need to check whether effort_mean is an integer that we need to map
    # to mean_output on the output scale.
    return map_to_rew


def map_efforts_to_rewards(list_of_efforts, map_to_rewards):
    """
    Maps effort responses to rewards.
    
    Args:
        list_of_efforts (list of int): effort responses to be mapped
        map_to_rewards (dictionary): maps effort responses to reward values (float)
    
    Returns:
        list (float): resulting reward values
    """
    rewards = []
    for e in list_of_efforts:
        rewards.append(map_to_rewards[e])
    
    return rewards
    
	
def pass_attention_checks(answer1, answer2):
    """
    Computes whether at least 1/2 attention checks were passed for a session.
    
    Args:
        answer1: given answer for attention check 1
        answer2: given answer for attention check 2
    
    Returns:
        bool: whether at least 1 of 2 attention checks were passed.
    """
    return answer1 in ATTENTION_CHECK_1_TRUE or answer2 in ATTENTION_CHECK_2_TRUE


def gather_data_post_sess_2(database_path, feat_to_select = [0, 1, 2, 3, 4, 6, 7],
                            excluded_ids = [[], []]):
    """
    Gathers data from sqlite database after session 2.
    
    Args:
        database_path (str): path to database
        feat_to_select (list of int): which of the state features to consider
        excluded_ids (list of list of str): list of IDs of people whose data we should 
                                            not use for each of the 2 sessions.
                                            
    Returns:
        list: (s, s_prime, a, r)-samples
        list: mean value for each feature in feat_to_select
        list: user ID for (s, s_prime, a, r)-sample
        float: mean value of the effort responses
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
        
        user_id_curr = data_db[row][0]
        
        try:
            # Get first state
            s0_arr = np.array([int(i) for i in data_db[row][20].split('|')])[feat_to_select]
            s0 = list(s0_arr)
            
        
            # Test data that is still in the database and that does not have
            # info for attention checks.
            if data_db[row][16] is None or data_db[row][17] is None or data_db[row][16] == ' ':
                passed_check_state = False
            
            # only if the user's ID is not in the list of users whose data we
            # should not use for the first session
            elif user_id_curr in excluded_ids[0]:
                passed_check_state = False
            
            else:
                # Attention check question answers for first session
                check_state_1 = [i for i in data_db[row][16].split('|')][0]
                check_state_2 = [i for i in data_db[row][17].split('|')][0]
                
                # make sure that there is actual data for the attention checks, i.e. not 
                # that the data saved in the database for this attention check is ''
                if check_state_1 in AW_LIKERT_SCALE and check_state_2 in AW_LIKERT_SCALE:
                    # Whether the first session has passed enough attention checks
                    passed_check_state = pass_attention_checks(int(check_state_1), int(check_state_2))
                # No true data for attention checks, so we do not use this sample.
                else:
                    passed_check_state = False
            
            # needed to later compute the mean values per feature
            if passed_check_state:
                all_states.append(s0_arr)
            
            # Check if the user has completed a second session
            if not data_db[row][21] == None:
                
                s1_arr = np.array([int(i) for i in data_db[row][21].split('|')])[feat_to_select]
                s1 = list(s1_arr)
            
                # Test data that is still in the database and that does not have
                # info for attention checks
                if data_db[row][16] is None or data_db[row][17] is None or data_db[row][16] == ' ':
                    passed_check_next_state = False
                # only if the user's ID is not in the list of users whose data we
                # should not use for the second session
                elif user_id_curr in excluded_ids[1]:
                    passed_check_next_state = False
                else:
                    # attention check answers for second session
                    check_next_1 = [i for i in data_db[row][16].split('|')][1]
                    check_next_2 = [i for i in data_db[row][17].split('|')][1]
                
                    # make sure that there is actual data for the attention checks, i.e. not 
                    # that the data saved in the database for this attention check is ''
                    if check_next_1 in AW_LIKERT_SCALE and check_next_2 in AW_LIKERT_SCALE:
                        # Whether the second session has passed enough attention checks
                        passed_check_next_state = pass_attention_checks(int(check_next_1), int(check_next_2))
                    
                    # No true data for attention checks, so we do not use this sample.
                    else:
                        passed_check_next_state = False
            
                # need to pass at least 1 out of 2 attention checks in each of the two sessions
                passed_attention_checks = passed_check_state and passed_check_next_state
                
                # the transition is not used if the attention check criteria are not met
                if passed_attention_checks:
                
                    a = [int(i) for i in data_db[row][25].split('|')][0]
                    r = [int(i) for i in data_db[row][7].split('|')][0]
                
                    data.append([s0, s1, a, r]) # save transition
                    user_ids.append(user_id_curr) # save corresponding user ID
                  
                # Even if the transition is not used, a state may still be used to compute 
                # mean values for the features
                if passed_check_next_state:
                    all_states.append(s1_arr)
            
        # Incomplete data for this user -> cannot be used.
        except Exception:
            print("Incomplete data for user " + data_db[row][0] + ".")

    all_states = np.array(all_states)
        
    # compute the mean value for each feature
    feat_means = np.mean(all_states, axis = 0)
    
    # convert features to binary features based on mean values
    for row in range(len(data)):
        data[row][0] = [1 if data[row][0][i] >= feat_means[i] else 0 for i in range(num_select)]
        data[row][1] = [1 if data[row][1][i] >= feat_means[i] else 0 for i in range(num_select)]
        
    # Compute the mean of the effort responses
    reward_mean = np.mean(np.array(data)[:, 3])
    
    return data, feat_means, user_ids, reward_mean


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
    user_ids_error = [] # users where something went wrong, i.e. some data but not all attention check data was saved
    
    session_index = session_num - 1
    
    print("... Checking attention checks for session ", session_num)
    
    # for each user
    for row in range(num_rows):
        
        session_error = False # whether something went wrong in a session, i.e. some data but not all attention check data was saved
    
        # Test data that is still in the database and that does not have
        # info for attention checks
        if data_db[row][16] is None or data_db[row][17] is None or data_db[row][16] == ' ':
            passed_check = False
        
        else:
            # Attention check question answers
            check_state_1 = [i for i in data_db[row][16].split('|')][session_index]
            check_state_2 = [i for i in data_db[row][17].split('|')][session_index]
            
            # make sure that there is actual data for the attention checks, i.e. not 
            # that the data saved in the database for this attention check is ''
            if check_state_1 in AW_LIKERT_SCALE and check_state_2 in AW_LIKERT_SCALE:
                # Whether the first session has passed enough attention checks
                passed_check = pass_attention_checks(int(check_state_1), int(check_state_2))
            # Something went wrong and not all attention check data has been collected.
            # So can't use this sample.
            else:
                passed_check = False
                session_error = True
       
        # save corresponding user ID
        if passed_check:
            user_ids_passed.append(data_db[row][0]) 
        else:
            if not session_error:
                user_ids_failed.append(data_db[row][0]) 
            else:
                user_ids_error.append(data_db[row][0])
        
    return user_ids_passed, user_ids_failed, user_ids_error


def policy_evaluation(observation_space_size, discount_factor, trans_func,
                      reward_func, policy, q_vals, 
                      update_tolerance = 0.000000001,
                      reward_dep_next_state = False):
    """
    Returns the q-values for a specific policy.
    
    Args:
        observation_space_size: number of observations
        discount_factor: discount factor of MDP
        trans_func: transition function
        reward_func: reward function
        policy: policy to compute q-values for
        update_tolerance: precision
        reward_dep_next_state (boolean, default: False): whether reward depends also on the next state
    """
    q_vals_new = deepcopy(q_vals)
    num_act = len(trans_func[0])
    
    update = 1 # max. change in Q-value in one iteration
   
    if not reward_dep_next_state: # reward does not also depend on next state
        while update > update_tolerance:
            update = 0
            for s in range(observation_space_size):
                for a in range(num_act):
                    q_vals_new[s, a] = reward_func[s, a] + discount_factor * sum([trans_func[s, a, s_prime] * q_vals[s_prime, int(policy[s_prime])] for s_prime in range(observation_space_size)])
                    update = max(update, abs(q_vals[s, a] - q_vals_new[s, a]))
            q_vals = deepcopy(q_vals_new)
    
    else: # reward does also depend on the next state
        while update > update_tolerance:
            update = 0
            for s in range(observation_space_size):
               
                for a in range(num_act):
                    q_vals_new[s, a] = sum([trans_func[s, a, s_prime] * (discount_factor * q_vals[s_prime, int(policy[s_prime])] + reward_func[s, a, s_prime]) for s_prime in range(observation_space_size)])
                    
                    update = max(update, abs(q_vals[s, a] - q_vals_new[s, a]))
            q_vals = deepcopy(q_vals_new)
        
    return q_vals_new


def get_Q_values_opt_policy(discount_factor, trans_func, reward_func,
                            reward_dep_next_state = False):
    """
    Returns the Q-values for each state under the optimal policy.
    
    Args:
        discount_factor: discount factor of MDP
        trans_func: transition function (dim.: num_states x num_actions x num_states)
        reward_func: reward function (dim.: num_states x num_actions)
        reward_dep_next_state (boolean, default: False): whether reward depends also on the next state
    """
    min_iterations = 100
    num_states = len(trans_func)
    num_act = len(trans_func[0])
    q_vals = np.zeros((num_states, num_act))

    policy = np.zeros(num_states)
    policy_new = np.ones(num_states)
    it = 0
    
    while (not np.array_equal(policy, policy_new)) or it < min_iterations:
        q_vals = policy_evaluation(num_states, discount_factor, 
                                   trans_func, reward_func, policy, q_vals,
                                   reward_dep_next_state = reward_dep_next_state)
        policy = policy_new
        policy_new = np.array([np.argmax(q_vals[s]) for s in range(num_states)])
        it += 1
    
    return q_vals, policy_new


def get_planning_reflection_answers(database_path, session_num):
    """
    Returns IDs and planning/reflection answers for a specific session.
    
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
    user_ids = [] # user IDs
    answers = [] # free-text responses given by users
    activities = [] # activities that people were assigned
    action_types = [] # persuasion types
    
    session_index = session_num - 1
    
    print("... Getting planning/reflection answers for session ", session_num)
    
    # for each user that has some saved data for this session
    for row in range(num_rows):
        
        user_id = data_db[row][0]
        
        try:
        
            # session_index goes from 0 to 4
            action_type = [i for i in data_db[row][25].split('|')][session_index]
            activity = [i for i in data_db[row][18].split('|')][session_index]
            
            # reflection answer
            if not action_type == "3":
                answer = data_db[row][29 + session_index]
            # planning answer
            else:
                if session_num < 5:
                    answer = data_db[row][3 + session_index]
                else:
                    answer = data_db[row][34]
                
            user_ids.append(user_id)
            answers.append(answer)
            activities.append(activity)
            action_types.append(action_type)
        
        except Exception:
            # Some but not all data has been saved in the database for this user.
            # This user's ID has already been collected in the list of error-IDs for 
            # a specific session and the user's rejection/approval will be examined
            # on a case-by-case basis. There is no need to save anything about this 
            # user here.
            print("Incomplete data for user " + user_id + ", session " + str(session_index) + ".")
    
    return user_ids, answers, activities, action_types
