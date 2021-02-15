"""
Selection of similarity variables.
"""
import numpy as np
import pandas as pd
import pickle
import itertools
import copy
import Utils as util
from itertools import islice
from random import randint
import random 

def partition (list_in, n):
    list_in = copy.deepcopy(list_in)
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

# load data. Data has <s, s', a, r> samples.
feat_to_select = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# TODO: use post-session-5 data in the end
data  = pd.read_csv('data_samples_post_sess_2.csv', converters={'s0': eval, 's1': eval})
data = data.values.tolist()

# TODO: choose features chosen after session 5 in the end
with open('Level_3_G_algorithm_chosen_features', 'rb') as f:
    feat_sel = pickle.load(f)

with open('IDs', 'rb') as f:
    user_ids = pickle.load(f)
num_people = len(user_ids)

# Create folds based on user IDs
# TODO: use 10 folds in the end
num_folds = 2
id_folds = partition(user_ids, num_folds)

num_act = 4
num_feat = len(feat_to_select)
num_samples = len(data)

# TODO: use all traits in the end
num_traits = 6 # Personality (5 dim.) and TTM-phase for PA 

# Settings for calculation of Q-values
q_num_iter = 100000 * num_samples # num_samples = num people after session 2
discount_factor = 0.85
alpha = 0.01

# TODO: to be replaced with actual data from Qualtricx
traits_orig = np.ones((num_samples, num_traits)) 
traits_orig[0, 0] = 5
traits_names = ['TTM_PA', 'P0', 'P1', 'P2', 'P3', 'P4']
df_traits = pd.DataFrame(traits_orig, columns=traits_names)

# Scale columns of traits to [0, 1]
for c in traits_names:
    df_traits[c] -= df_traits[c].min()
    df_traits[c] /= df_traits[c].max()
    
improv_threshold = 0.001
improv = 1

trait_sel = []

while improv > improv_threshold: # while adding more traits leads to improvement

    # not yet selected features
    trait_not_sel = [i for i in traits_names if not i in trait_sel]
    
    for trait in trait_not_sel: # for each not yet selected feature
    
        traits = df_traits[trait_sel+[trait]].to_numpy() # selected + this not yet selected feature
    
        for fold in range(num_folds): # for each fold of user IDs
        
            fold_test_people = id_folds[fold]
            num_test_people = len(fold_test_people)
            fold_train_people = [i for i in user_ids if not i in fold_test_people]
            num_train_people = len(fold_train_people)
        
            # for each test person
            for p1 in fold_test_people:
                
                data_p = copy.deepcopy(data)
                
                d_E = np.zeros(num_train_people)
                
                # compute Euclidean distances based on traits for each sample
                for p2 in fold_train_people:
                    
                    d_E[p2] = np.linalg.norm(traits[p1] - traits[p2])
                
                sum_E = sum(d_E) # sum of Euclidean distances
                
                # increase frequency of samples based on Euclidean distances
                for p2 in range(num_samples):
                    weight = int(round(d_E[p2]/sum_E, 3) * 1000) # increase sample size by factor of 1000
                    
                    for w in range(weight):
                        data_p.append(data[p2])
                        
                num_samples_p = len(data_p)
               
                abstract_states = [list(i) for i in itertools.product([0, 1], repeat = num_feat)]
            
                # Compute transition function and reward function
                trans_func = np.zeros((int(2 ** num_feat), num_act, int(2 ** num_feat)))
                reward_func = np.zeros((int(2 ** num_feat), num_act))
                reward_func_count = np.zeros((int(2 ** num_feat), num_act))
                for s_ind, s in enumerate(abstract_states):
                    for data_index in range(num_samples):
                        if list(np.take(np.array(data[data_index][0]), feat_sel)) == s:
                            trans_func[s_ind, data[data_index][2], abstract_states.index(list(np.take(data[data_index][1], feat_sel)))] += 1
                            r = int(data[data_index][3])
                            if r == 0:
                                r = -1
                            reward_func[s_ind, data[data_index][2]] += r
                            reward_func_count[s_ind, data[data_index][2]] += 1
                   
                    # Normalize
                    for a in range(num_act):
                        summed = sum(trans_func[s_ind, a])
                        if summed > 0:
                            trans_func[s_ind, a] /= summed
                        if reward_func_count[s_ind, a] > 0:
                            reward_func[s_ind, a] /= reward_func_count[s_ind, a]
                
                # Value iteration        
                q_values_exact, _ = util.get_Q_values_opt_policy(discount_factor, trans_func, reward_func)
                opt_policy = [[[[a for a in range(num_act) if q_values_exact[abstract_states.index([i, j, k])][a] == max(q_values_exact[abstract_states.index([i, j, k])])] for k in range(2)] for j in range(2)] for i in range(2)]
            
            