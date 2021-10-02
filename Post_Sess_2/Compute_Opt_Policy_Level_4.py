'''
Computations for level 4 of algorithm complexity.
Considers TTM-stage for becoming more physically active and big-5 personality
for computing the similarity of people.
'''


import copy
import itertools
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import Utils as util


def approx_dynamics(data_p, abstract_states, num_feat, num_act, feat_sel):
    """Compute the optimal policy for level 4 of algorithm complexity.

    Args:
        data_p (list): <s0, s1, a, r>-samples.
        abstract_states (list): Abstract states.
        num_feat (int): Number of features used.
        num_act (int): Number of actions.
        feat_sel (list): Selected features

    Returns:
        np-array: reward function.
        np-array: transition function.

    """
    
    num_samples_p = len(data_p)
    
    # Compute approximate transition function and reward function
    trans_func = np.zeros((int(2 ** num_feat), num_act, int(2 ** num_feat)))
    reward_func = np.zeros((int(2 ** num_feat), num_act, int(2 ** num_feat)))
    reward_func_count = np.zeros((int(2 ** num_feat), num_act, int(2 ** num_feat)))
    
    for s_ind, s in enumerate(abstract_states):
        for data_index in range(num_samples_p):  # for each data sample
            if list(np.take(np.array(data_p[data_index][0]), feat_sel)) == s:
                
                a = data_p[data_index][2]  # action
                r = data_p[data_index][3]
                next_state_index = abstract_states.index(list(np.take(data_p[data_index][1], feat_sel)))
                
                trans_func[s_ind, a, next_state_index] += 1
                reward_func[s_ind, a, next_state_index] += r
                reward_func_count[s_ind, a, next_state_index] += 1
       
        # Normalize reward and transition functions
        for a in range(num_act):
            summed = sum(trans_func[s_ind, a])
            if summed > 0:
                trans_func[s_ind, a] /= summed
            # If we have no data on a state-action combination, 
            # we assume equal probability of transitioning to each other state
            else:
                trans_func[s_ind, a] = np.ones(int(2 ** num_feat)) / (2 ** num_feat)
            
            for s_prime_ind in range(len(abstract_states)):
                if reward_func_count[s_ind, a, s_prime_ind] > 0:
                    reward_func[s_ind, a, s_prime_ind] /= reward_func_count[s_ind, a, s_prime_ind]
        
    return reward_func, trans_func


def compute_opt_policy_level_4(data_in, effort_mean, feat_sel, user_ids,
                               user_ids_assigned_group_4, traits,
                               traits_ids, num_act=5, discount_factor = 0.85):
    """Compute the optimal policy for level 4 of algorithm complexity.

    Args:
        data_in (list): List with samples of the form <s0, s1, a, r>.
        effort_mean (float): Mean value of effort responses.
        feat_sel (list): Selected features.
        user_ids (list): IDs of users corresponding to samples in data.
        user_ids_assigned_group_4 (list): IDs of users for whom we need to compute policies.
        traits (np-array): Traits for each person to use as basis for similarity.
        traits_ids (list): IDs corresponding to traits.
        num_act (int): Number of possible actions.
        discount_factor (float): Discount factor for Q-value computation

    Returns:
        dict: Optimal policy for each person in group 4.

    """
    
    data = copy.deepcopy(data_in)
    
    num_assigned = len(user_ids_assigned_group_4)  # number of people for whom we need to calculate policies
    num_feat = len(feat_sel)  # number of selected features that we now consider
    num_samples = len(data)
    
    # All effort responses
    list_of_efforts = list(np.array(data)[:, 3].astype(int))
    
    # Map effort responses to rewards from -1 to 1, with the mean mapped to 0.
    map_to_rewards = util.get_map_effort_reward(effort_mean, 
                                                output_lower_bound = -1, 
                                                output_upper_bound = 1, 
                                                input_lower_bound = 0, 
                                                input_upper_bound = 10)
    reward_list = util.map_efforts_to_rewards(list_of_efforts, map_to_rewards)
    # now write these obtained reward values into "data"
    for i in range(len(reward_list)):
        data[i][3] = reward_list[i]
    
    opt_policies = {}

    # For each person for whom we need to calculate a policy, i.e. all people in group 4.
    # For people who are assigned to group 4 later on this means that their own data
    # sample is not used in the computation of their policy.
    # This is so that the number of samples used for computing policies does not
    # change throughout the experiment.
    for p1 in range(num_assigned):
        
        print("Person index from Group 4 in assignment.csv:", p1)
        
        # Get index of traits for this person
        trait_index_p1 = traits_ids.index(user_ids_assigned_group_4[p1])
        
        data_p = copy.deepcopy(data)
        
        # Save Euclidean distances of traits
        d_E = np.zeros(num_samples)
        
        # Compute Euclidean distances based on traits for each sample
        for p2 in range(num_samples):
            
            # Get index of traits for this person
            trait_index_p2 = traits_ids.index(user_ids[p2])
            
            d_E[p2] = np.linalg.norm(traits[trait_index_p1] - traits[trait_index_p2])
            
        # We use min-max scaling, and afterwards take the absolute value to 
        # inverse the weights. The goal is that people with a low Euclidean 
        # distance between their trait vectors have a high similarity.
        scaler = MinMaxScaler(feature_range = (-1, 0))
        scaler.fit([[d_E_val] for d_E_val in d_E])
        d_E = scaler.transform([[d_E_val] for d_E_val in d_E])
        
        # We take the absolute value to reverse the weights compared to 
        # the Euclidean distance, i.e. a large Euclidean distance should 
        # mean a low weight
        d_E = [np.abs(d_E_val[0]) for d_E_val in d_E]
        
        # Need to compute the sum since we want all weights to add up to 1 
        sum_E = sum(d_E)
        
        # Increase frequency of samples based on Euclidean distances
        for p2 in range(num_samples):
            weight = int(round(d_E[p2]/sum_E, 4) * 10000)  # increase sample size by a factor
            # Note that if a sample's weight is 0, we still keep the single sample
            # that we already had in our dataset. So we throw away no data.
            # We have weight - 1 here since we already have 1 sample in the dataset
            for w in range(weight - 1):
                data_p.append(data[p2])
            
        
        abstract_states = [list(i) for i in itertools.product([0, 1], repeat = num_feat)]
    
        # Approximate reward and transition functions
        reward_func, trans_func = approx_dynamics(data_p, abstract_states,
                                                  num_feat, num_act, feat_sel)
        
        # Value iteration; reward depends also on next state
        q_values_exact, _ = util.get_Q_values_opt_policy(discount_factor, trans_func,
                                                         reward_func, 
                                                         reward_dep_next_state = True)
        
        # Optimal policy, can give multiple best actions in a state
        opt_policy = [[[[a for a in range(num_act) if q_values_exact[abstract_states.index([i, j, k])][a] == max(q_values_exact[abstract_states.index([i, j, k])])] for k in range(2)] for j in range(2)] for i in range(2)]
        
        # Save computed policy for this person
        opt_policies[user_ids_assigned_group_4[p1]] = opt_policy
        
    return opt_policies
    
    
if __name__ == "__main__":
    
    DISCOUNT_FACTOR  = 0.85  # For computation of Q-values
    NUM_ACTIONS = 5  # We have 5 persuasion types
    
    # load data. Data has <s, s', a, r> samples.
    data  = pd.read_csv('W:/staff-umbrella/perfectfit/Exp0/Final_Algorithms/2021_05_27_1401_data_samples_post_sess_2.csv', 
                        converters={'s0': eval, 's1': eval})
    data = data.values.tolist()
    
    # Mean value of effort responses
    with open("W:/staff-umbrella/perfectfit/Exp0/Final_Algorithms/2021_05_27_1401_Post_Sess_2_Effort_Mean", "rb") as f:
        effort_mean = pickle.load(f)
        
    # features chosen for level 3
    with open('W:/staff-umbrella/perfectfit/Exp0/Final_Algorithms/Level_3_G_algorithm_chosen_features', 'rb') as f:
        feat_sel = pickle.load(f)
    
    # IDs of users correponding to data samples
    with open('W:/staff-umbrella/perfectfit/Exp0/Final_Algorithms/2021_05_27_1401_IDs', 'rb') as f:
        user_ids = pickle.load(f)
        
    # Get group assignments 
    df_group_ass = pd.read_csv("W:/staff-umbrella/perfectfit/Exp0/assignment.csv", 
                               dtype={'ID':'string'})
    # Only consider people assigned to group 4 (index 3)
    df_group_ass_group_4 = df_group_ass[df_group_ass["Group"].isin([3])]
    user_ids_assigned_group_4 = df_group_ass_group_4['ID'].tolist()
    
    # Load similarity variables for all people
    traits = pd.read_csv("W:/staff-umbrella/perfectfit/Exp0/Extract_Data/pers_PA-TTM_gender_MergedAll_1859.csv")
    traits_ids = traits['PROLIFIC_PID'].tolist()
    traits = traits[['PA-TTM', 'Extraversion', 'Agreeableness', 'Conscientiousness', 'ES', 'OE']].to_numpy()
    
    # Compute optimal policies for people in group 4
    # We have 5 persuasion types and 6 traits (5 personality dimensions
    # and TTM-stage for PA)
    opt_policies = compute_opt_policy_level_4(data, effort_mean, feat_sel, 
                                              user_ids,
                                              user_ids_assigned_group_4, 
                                              traits, 
                                              traits_ids, 
                                              num_act = NUM_ACTIONS,
                                              discount_factor = DISCOUNT_FACTOR)
    
    with open('Level_4_Optimal_Policy', 'wb') as f:
        pickle.dump(opt_policies, f)