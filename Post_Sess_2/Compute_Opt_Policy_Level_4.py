'''
Computations for Group 4.
Needs to be run on server (i.e. not locally) as it requires user IDs.
Considers TTM-stage for becoming more physically active and big-5 personality
for computing the similarity of people.
'''
import numpy as np
import pickle
import pandas as pd
import copy
import Utils as util
import itertools

# load data. Data has <s, s', a, r> samples.
data  = pd.read_csv('data_samples_post_sess_2.csv', converters={'s0': eval, 's1': eval})
data = data.values.tolist()

# All effort responses
list_of_efforts = list(np.array(data)[:, 3].astype(int))
# Mean value of effort responses
with open("Post_Sess_2_Effort_Mean", "rb") as f:
    effort_mean = pickle.load(f)
# Map effort responses to rewards from -1 to 1, with the mean mapped to 0.
map_to_rewards = util.get_map_effort_reward(effort_mean, output_lower_bound = -1, 
                                            output_upper_bound = 1, 
                                            input_lower_bound = 0, 
                                            input_upper_bound = 10)
reward_list = util.map_efforts_to_rewards(list_of_efforts, map_to_rewards)
# now write these obtained reward values into "data"
for i in range(len(reward_list)):
    data[i][3] = reward_list[i]

with open('Level_3_G_algorithm_chosen_features', 'rb') as f:
    feat_sel = pickle.load(f)

with open('IDs', 'rb') as f:
    user_ids = pickle.load(f)

# Get group assignments 
# TODO: adapt path in the end
df_group_ass = pd.read_csv("c:/users/nele2/CAS/assignment.csv", dtype={'ID':'string'})

num_act = 5 # number of actions
num_feat = len(feat_sel) # number of selected features that we now consider
num_samples = len(data)
num_traits = 6 # Personality (5 dim.) and TTM-phase for PA

# Settings for calculation of Q-values
q_num_iter = 100000 * num_samples # num_samples = num people after session 2
discount_factor = 0.85
alpha = 0.01

# TODO: add correct file path
traits = pd.read_csv("W:/staff-umbrella/perfectfit/Exp0/Extract_Data/Pilot2_pers_PA-TTM_gender.csv")
traits_ids = traits['PROLIFIC_PID'].tolist()
traits = traits[['PA-TTM', 'Extraversion', 'Agreeableness', 'Conscientiousness', 'ES', 'OE']]
traits = traits.to_numpy()

opt_policies = {}

# for each person
for p1 in range(num_samples):
    
    print("Person index:", p1)
    
    # TODO: change to only people that belong to Group 4 (group numbers go from 0 to 3)
    # Right now we have more groups here due to the limited test samples we have.
    if str(df_group_ass[df_group_ass['ID'] == user_ids[p1]]["Group"].tolist()[0]) in ["3", "1", "2", "5"]:
    
        # Get index of traits for this person
        trait_index_p1 = traits_ids.index(user_ids[p1])
        
        data_p = copy.deepcopy(data)
        
        # Save Euclidean distances of traits
        d_E = np.zeros(num_samples)
        
        # compute Euclidean distances based on traits for each sample
        for p2 in range(num_samples):
            
            # Get index of traits for this person
            trait_index_p2 = traits_ids.index(user_ids[p2])
            
            d_E[p2] = np.linalg.norm(traits[trait_index_p1] - traits[trait_index_p2])
            
        sum_E = sum(d_E) # sum of Euclidean distances
        
        # Intermediate weights of samples based on Euclidean distances
        # So if Euclidean distance is 0, then now the weight is 1.
        # If the Euclidean distance is higher than 0, then now the weight is less than 1.
        for p2 in range(num_samples):
            d_E[p2] = (sum_E - d_E[p2]) / sum_E
        
    
        # need to compute the sum again since we want all weights to add up to 1 in the end
        sum_E = sum(d_E)
        
        # increase frequency of samples based on Euclidean distances
        for p2 in range(num_samples):
            weight = int(round(d_E[p2]/sum_E, 3) * 1000) # increase sample size by factor of 1000
          
            # Note that if a sample's weight is 0, we still keep the single sample
            # that we already had in our dataset. So we throw away no data.
            # We have weight - 1 here since we already have 1 sample in the dataset
            for w in range(weight - 1):
                data_p.append(data[p2])
            
        num_samples_p = len(data_p)
        
        '''
        # Compute Q-values
        q_values = np.zeros((2, 2, 2, num_act))
        for t in range(q_num_iter):
            data_index = np.random.randint(0, num_samples_p)
            s = np.take(np.array(data_p[data_index][0]), feat_sel)
            s_prime = np.take(np.array(data_p[data_index][1]), feat_sel)
            a = data_p[data_index][2]
            r = int(data_p[data_index][3])
            if r == 0:
                r = -1
                  
            # TD Update 
            best_next_action = np.argmax(q_values[s_prime[0], s_prime[1], s_prime[2]])     
            td_target = r + discount_factor * q_values[s_prime[0], s_prime[1], s_prime[2], best_next_action] 
            td_delta = td_target - q_values[s[0], s[1], s[2], a] 
            q_values[s[0], s[1], s[2], a]  += alpha * td_delta 
        
        opt_policy = [[[[a for a in range(num_act) if q_values[i, j, k, a] == max(q_values[i, j, k])] for k in range(2)] for j in range(2)] for i in range(2)]
        '''
        abstract_states = [list(i) for i in itertools.product([0, 1], repeat = 3)]

        # Compute approximate transition function and reward function
        trans_func = np.zeros((int(2 ** num_feat), num_act, int(2 ** num_feat)))
        reward_func = np.zeros((int(2 ** num_feat), num_act))
        reward_func_count = np.zeros((int(2 ** num_feat), num_act))
        for s_ind, s in enumerate(abstract_states):
            for data_index in range(num_samples_p):
                if list(np.take(np.array(data_p[data_index][0]), feat_sel)) == s:
                    trans_func[s_ind, data_p[data_index][2], abstract_states.index(list(np.take(data_p[data_index][1], feat_sel)))] += 1
                    r = data_p[data_index][3]
                    reward_func[s_ind, data_p[data_index][2]] += r
                    reward_func_count[s_ind, data_p[data_index][2]] += 1
           
            # Normalize
            for a in range(num_act):
                summed = sum(trans_func[s_ind, a])
                if summed > 0:
                    trans_func[s_ind, a] /= summed
                # if we have no data on a state-action combination, we assume equal probability of transitioning to each other state
                else:
                    trans_func[s_ind, a] = np.ones(int(2 ** num_feat)) / (2 ** num_feat)
                if reward_func_count[s_ind, a] > 0:
                    reward_func[s_ind, a] /= reward_func_count[s_ind, a]
        
        # Value iteration        
        q_values_exact, _ = util.get_Q_values_opt_policy(discount_factor, trans_func, reward_func)
        opt_policy = [[[[a for a in range(num_act) if q_values_exact[abstract_states.index([i, j, k])][a] == max(q_values_exact[abstract_states.index([i, j, k])])] for k in range(2)] for j in range(2)] for i in range(2)]

        opt_policies[user_ids[p1]] = opt_policy
    
with open('Level_4_Optimal_Policy', 'wb') as f:
    pickle.dump(opt_policies, f)