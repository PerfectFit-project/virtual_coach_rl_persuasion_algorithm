'''
Computations for Group 4.
'''
import numpy as np
import pickle
import pandas as pd
import copy

# load data. Data has <s, s', a, r> samples.
feat_to_select = [0, 1, 2, 3, 4, 6, 7]
data  = pd.read_csv('data_samples_post_sess_2.csv', converters={'s0': eval, 's1': eval})
data = data.values.tolist()

with open('Level_3_G_algorithm_chosen_features', 'rb') as f:
    feat_sel = pickle.load(f)

with open('IDs', 'rb') as f:
    user_ids = pickle.load(f)

num_act = 4
num_feat = len(feat_to_select)
num_samples = len(data)
num_traits = 6 # Personality and TTM-phase for PA

# Settings for calculation of Q-values
q_num_iter = 1000 * num_samples # num_samples = num people after session 2
discount_factor = 0.85
alpha = 0.01

# to be replaced with actual data from Qualtricx
traits = np.ones((num_samples, num_traits)) 
traits[0, 0] = 5

opt_policies = {}

for p1 in range(num_samples):
    
    data_p = copy.deepcopy(data)
    
    d_E = np.zeros(num_samples)
    
    # compute Euclidean distances based on traits for each sample
    for p2 in range(num_samples):
        
        d_E[p2] = np.linalg.norm(traits[p1] - traits[p2])
    
    sum_E = sum(d_E) # sum of Euclidean distances
    
    # increase frequency of samples based on Euclidean distances
    for p2 in range(num_samples):
        weight = int(round(d_E[p2]/sum_E, 3) * 1000) # increase sample size by factor of 1000
        
        for w in range(weight):
            data_p.append(data[p2])
            
    num_samples_p = len(data_p)
    
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
    
    opt_policies[user_ids[p1]] = opt_policy
    
with open('Level_4_Optimal_Policy', 'wb') as f:
    pickle.dump(opt_policies, f)