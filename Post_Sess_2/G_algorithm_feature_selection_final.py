'''
Final version of feature selection based on G-algorithm for experiment.
To be run after session 2
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
q_num_iter = 1000 * num_samples # num_samples = num people after session 2
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
        r = int(data[data_index][3])
        if r == 0:
            r = -1
                   
        # TD Update 
        best_next_action = np.argmax(q_values[f, s_prime])     
        td_target = r + discount_factor * q_values[f, s_prime][best_next_action] 
        td_delta = td_target - q_values[f][s][a] 
        q_values[f][s][a] += alpha * td_delta 
    
    # t-test
    t_tests[f] = stats.ttest_ind(q_values[f][0], q_values[f][1])[0]

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
    q_values_2 = np.zeros((num_blocks, num_feat_not_sel, 2, num_act))
    t_tests_2 = np.zeros((num_blocks, num_feat_not_sel))
    
    for b_ind, block in enumerate(blocks): # for each block
        for f_ind, f in enumerate(feat_not_sel): # for each not yet selected feature
            for t in range(q_num_iter):
                
                data_index = np.random.randint(0, num_samples) # data sample
                s = data[data_index][0][f]
                s_b = np.take(data[data_index][0], feat_sel)
                s_prime = data[data_index][1][f]
                s_prime_b = np.take(data[data_index][1], feat_sel)
                
                # both s and s' must be in the current block
                if list(s_b) == block and list(s_prime_b) == block:
                    
                    a = data[data_index][2]
                    r = int(data[data_index][3])
                    if r == 0:
                        r = -1
                               
                    # TD Update 
                    best_next_action = np.argmax(q_values_2[b_ind, f_ind, s_prime])     
                    td_target = r + discount_factor * q_values_2[b_ind, f_ind, s_prime, best_next_action] 
                    td_delta = td_target - q_values_2[b_ind, f_ind, s, a] 
                    q_values_2[b_ind][f_ind][s][a] += alpha * td_delta 
            
            # t-test
            t_tests_2[b_ind, f_ind] = stats.ttest_ind(q_values_2[b_ind][f_ind][0], q_values_2[b_ind][f_ind][1])[0]
    
    # Select next feature
    feat_sel, feat_sel_criteria = util.feat_sel_num_blocks_avg_p_val(feat_not_sel, num_feat_not_sel, 
                                                                     blocks, 
                                                                     t_tests_2, feat_sel,
                                                                     feat_sel_criteria)
        
    print("Feature selected:", feat_sel[-1])
    print("Criterion:", feat_sel_criteria[-1])

# Store selected features
with open('Level_3_G_algorithm_chosen_features', 'wb') as f:
    pickle.dump(feat_sel, f)
with open("Level_3_G_algorithm_chosen_features_criteria", 'wb') as f:
    pickle.dump(feat_sel_criteria, f)
    
# Compute Q-values
q_values = np.zeros((2, 2, 2, num_act))
for t in range(q_num_iter):
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

with open('Level_3_Optimal_Policy', 'wb') as f:
    pickle.dump(opt_policy, f)