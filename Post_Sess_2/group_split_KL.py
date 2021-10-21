'''
Split of data into 4 groups based on Kullback-Leibler distance.
Consider Big-5 personality, gender (gender identity in Prolific)
 and TTM-stage for physical activity, as well as the effort response from Session 2.
'''


import math
import numpy as np
import pandas as pd
import pickle


def gau_kl(pm, pv, qm, qv):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.
    
    Formula: https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians.
    Code: http://www.cs.cmu.edu/~chanwook/MySoftware/rm1_Spk-by-Spk_MLLR/rm1_PNCC_MLLR_1/rm1/python/sphinx/divergence.py.
    """
    # Determinants of diagonal covariances pv, qv
    dpv = np.linalg.det(pv)
    dqv = np.linalg.det(qv)
    # if dqv = 0, we need to divide by zero later on inside the logarithm.
    # This can happen if one group has no variance in one of the features, 
    # most likely in the TTM-phase-feature.
    # This is unlikely to happen in the actual experiment, albeit it happens
    # with the pilot data.
    # Since we do not want assignments with no variance in a feature in a group,
    # we just return infinity then. The assignment will then not be chosen,
    # as we pick the assignment with minimum avg. KL-divergence.
    # So just that we are aware of it happening, we print a message out here.
    if dqv == 0:
        print("Negative determinant of second covariance matrix.")
        return np.inf
    # Inverse of diagonal covariance qv
    iqv = 1./qv
    # Set non-diagonal entries to 0 rather than nan
    iqv = np.tril(np.triu(iqv, k=0), k=0)
    # Difference between means pm, qm
    diff = qm - pm
    return (0.5 *
            (np.log(dqv / dpv)            # log |\Sigma_q| / |\Sigma_p|
             + np.matrix.trace(iqv * pv)        # + tr(\Sigma_q^{-1} * \Sigma_p)
             + np.dot(diff, np.dot(iqv, diff)) # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
             - len(pm)))                     # - N


num_rep = 10000 # number of splits to test, 10,000
num_groups = 4 # want 4 groups


# we might have already assigned some people previously, e.g.
# when we need to get more people later on because of dropout etc.
# TODO: adapt if we have already previously assigned people
previous_assignment = True
if previous_assignment:
    df_assignment_prev = pd.read_csv("assignment_prev.csv")


# Load people who have passed the second session.
# Only want to assign those people to groups.
people_passed_session_2 = pd.read_csv("session2_answers_passed_p_ids.csv")
# People with nonsensical planning/reflection answers in session 2
people_nonsens_session_2 = pd.read_csv("session2_nonsensical_answers.csv")


# Get data on gender, PA-TTM phase and personality
traits = pd.read_csv('pers_PA-TTM_gender.csv') 
df_data_orig = traits[['PROLIFIC_PID', 'PA-TTM', 'Gender identity', 'Extraversion', 'Agreeableness', 'Conscientiousness', 'ES', 'OE']] 


# We also need to load data about the effort responses that people gave in session 2
# This data was stored in the database.
data_database  = pd.read_csv('data_samples_post_sess_2.csv', 
                             converters={'s0': eval, 's1': eval})
data_database = data_database.values.tolist()
# Load the corresponding IDs of people
with open("IDs", "rb") as f:
    ids_database = pickle.load(f)


# only consider people who have passed session 2 and did 
# not give nonsensical planning/reflection answers in session 2
df_data_orig = df_data_orig[df_data_orig['PROLIFIC_PID'].isin(people_passed_session_2['Prolific_ID'].tolist())]
df_data_orig = df_data_orig[~df_data_orig['PROLIFIC_PID'].isin(people_nonsens_session_2['Prolific_ID'].tolist())]
df_data_orig = df_data_orig.reset_index() # make sure there are consecutive indices


if previous_assignment:
    df_data_orig_incl_prev = df_data_orig.copy(deep = True) # including the people we have previously assigned to groups already
    # remove previously assigned people from this dataframe
    df_data_orig = df_data_orig[~df_data_orig['PROLIFIC_PID'].isin(df_assignment_prev['ID'].tolist())]
    df_data_orig = df_data_orig.reset_index() # make sure there are consecutive indices


num_people = len(df_data_orig)


# Now we need to add people's effort response in session 2 to the dataframe
# For that we first need to also scale the effort responses to [0, 1]
# So we divide by the highest possible effort response, which is 10.
# We do not first need to subtract 1, because the lowest possible value is
# already 0 (and not 1).
list_of_efforts = np.array(data_database)[:, 3].astype(int)/10.0
df_data_orig["Effort_Session2"] = [[list_of_efforts[i] for i in range(len(list_of_efforts)) if ids_database[i] == df_data_orig.loc[j, "PROLIFIC_PID"]][0] for j in range(num_people)]
if previous_assignment:
    df_data_orig_incl_prev["Effort_Session2"] = [[list_of_efforts[i] for i in range(len(list_of_efforts)) if ids_database[i] == df_data_orig_incl_prev.loc[j, "PROLIFIC_PID"]][0] for j in range(len(df_data_orig_incl_prev))]


# Create random assignments to groups
assignments = []
assignments_means = []
assignments_covs = []
for rep in range(num_rep): # for each repetition

    # Print progress
    if rep % 100 == 0:
        print("Rep", rep)
    
    # since we drop rows repeatedly during this repetition, we need a deep copy
    df_data = df_data_orig.copy(deep = True)
    
    parts = []
    means = []
    covs = []
    parts_ids = []
    
    for g in range(num_groups): # sample groups
        
        # split by gender and remove gender column so it does not factor into Gaussian
        df_data_gen0 = df_data[df_data['Gender identity'] == 0][['PROLIFIC_PID', 'PA-TTM', 'Extraversion', 'Agreeableness', 'Conscientiousness', 'ES', 'OE', "Effort_Session2"]]
        df_data_gen05 = df_data[df_data['Gender identity'] == 0.5][['PROLIFIC_PID', 'PA-TTM', 'Extraversion', 'Agreeableness', 'Conscientiousness', 'ES', 'OE', "Effort_Session2"]]
        df_data_gen1 = df_data[df_data['Gender identity'] == 1][['PROLIFIC_PID', 'PA-TTM', 'Extraversion', 'Agreeableness', 'Conscientiousness', 'ES', 'OE', "Effort_Session2"]]
        
        # sample per gender category
        sample_gen0 = df_data_gen0.sample(frac = 1.0/(num_groups - g))
        sample_gen05 = df_data_gen05.sample(frac = 1.0/(num_groups - g))
        sample_gen1 = df_data_gen1.sample(frac = 1.0/(num_groups - g))
        
        # get traits and ids
        sample_gen0_traits = sample_gen0[['PA-TTM', 'Extraversion', 'Agreeableness', 'Conscientiousness', 'ES', 'OE', "Effort_Session2"]]
        sample_gen05_traits = sample_gen05[['PA-TTM', 'Extraversion', 'Agreeableness', 'Conscientiousness', 'ES', 'OE', "Effort_Session2"]]
        sample_gen1_traits = sample_gen1[['PA-TTM', 'Extraversion', 'Agreeableness', 'Conscientiousness', 'ES', 'OE', "Effort_Session2"]]
        merge_gen0051 = pd.concat([sample_gen0_traits, sample_gen05_traits, sample_gen1_traits])
        parts.append(merge_gen0051)
        sample_gen0_ids = sample_gen0['PROLIFIC_PID'].tolist()
        sample_gen05_ids = sample_gen05['PROLIFIC_PID'].tolist()
        sample_gen1_ids = sample_gen1['PROLIFIC_PID'].tolist()
        parts_ids.append(sample_gen0_ids + sample_gen05_ids + sample_gen1_ids)
            
        # Last group, i.e. finished basic sampling
        # But there might be some people left.
        if g == num_groups - 1:
            
            # First drop the people we already sampled to be in the last group
            df_data = df_data.drop(sample_gen0.index)
            df_data = df_data.drop(sample_gen05.index)
            df_data = df_data.drop(sample_gen1.index)
            
            # check if there are people left
            # If yes, we assign them to groups without first splitting by gender.
            if len(df_data) > 0:
                for g_rest in range(num_groups):
                    sample_rest = df_data(frac = 1/(4-g_rest))
                    
                    # get IDs and traits from sample
                    sample_rest_traits = sample_rest[['PA-TTM', 'Extraversion', 'Agreeableness', 'Conscientiousness', 'ES', 'OE', "Effort_Session2"]]
                    parts[g] = pd.concat([parts[g], sample_rest_traits])
                    sample_rest_ids = sample_rest['PROLIFIC_PID'].tolist()
                    parts_ids[g] += sample_rest_ids
                    
                    # drop from dataframe
                    df_data = df_data.drop(sample_rest.index)
                    
        else:          
            # drop indices that have already been selected before we sample for the next group
            df_data = df_data.drop(sample_gen0.index)
            df_data = df_data.drop(sample_gen05.index)
            df_data = df_data.drop(sample_gen1.index)
        
    # Compute means and covariance matrices for each group
    for g in range(num_groups):
        
        if previous_assignment:
            # get IDs of people previously assigned to the current group
            prev_group_ids = df_assignment_prev[df_assignment_prev['Group'] == g][['ID']]
            if len(prev_group_ids) > 0:
                # get the corresponding traits (without gender)
                prev_traits = df_data_orig_incl_prev[df_data_orig_incl_prev['PROLIFIC_PID'].isin(prev_group_ids['ID'].tolist())][['PA-TTM', 'Extraversion', 'Agreeableness', 'Conscientiousness', 'ES', 'OE', 'Effort_Session2']]
                # merge the dataframes for new and previously assigned people for this group
                # The previously assigned people need to be considered when estimating the means and covariances
                parts[g] = pd.concat([parts[g], prev_traits])
                parts_ids[g] += prev_group_ids['ID'].tolist()
        
        # We get an error here if one of the groups has only 1 person
        means.append(np.mean(parts[g].values, axis = 0)) # estimate mean of Gaussian
        covs.append(np.cov(parts[g].values, rowvar = 0)) # estimate covariance matrix
        covs[g] = np.tril(np.triu(covs[g], k=0), k=0) # turn into diagonal cov matrix
        print(covs[g])
        
    assignments.append(parts_ids)
    assignments_means.append(means)
    assignments_covs.append(covs)


# Compute symmetric measure based on KL-divergence
# i.e. we compute Jeffrey's distance = the sum of both directions
all_kl = []  
for rep in range(num_rep):
    kl_rep = []
    for g1 in range(num_groups - 1):
        for g2 in range(g1 + 1, num_groups):
            kl1 = gau_kl(assignments_means[rep][g1], assignments_covs[rep][g1],
                        assignments_means[rep][g2], assignments_covs[rep][g2])
            kl2 = gau_kl(assignments_means[rep][g2], assignments_covs[rep][g2],
                        assignments_means[rep][g1], assignments_covs[rep][g1])
            kl_rep.append(kl1 + kl2) # we do not take the mean; this is Jeffrey's distance, i.e. sum of both directions
    all_kl.append(kl_rep)


# Compute mean KL-divergences for each repetition
kl_means = [np.mean(all_kl[i]) for i in range(num_rep)]
# Convert nan to infinity so that nan-assignments are not seen as those with lowest 
# mean KL-divergence
# We can get nan-values if all participants in one group have the same value for one feature.
kl_means = [i if not math.isnan(i) else np.inf for i in kl_means]


# Find repetition with lowest mean KL-divergence
best_rep = np.argmin(kl_means)
print("Chosen assignment:", best_rep, "with avg. KL-div.", 
      np.round(np.min(kl_means), 3))


# Get corresponding assignments to groups
indices = assignments[best_rep]
all_ids = [item for sublist in indices for item in sublist] # all IDs, incl. previously assigned ones


# Get chosen group number per person
groups = [[j for j in range(num_groups) if i in indices[j]] for i in all_ids]
groups = [j[0] for j in groups]


# combine groups and IDs
data_final = np.transpose([all_ids, groups])


# Save chosen group number and ID per person
# We also save the IDs of previously assigned people again, as we might still
# need to retrieve them during the experiment, and also need to know them later.
df_assignment = pd.DataFrame(data_final, columns = ['ID', 'Group']) 
df_assignment.to_csv("assignment.csv",
                     index = True)
