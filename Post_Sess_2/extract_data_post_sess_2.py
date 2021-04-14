'''
Script to extract data samples (i.e. <s, a, r, s'>) from database after session 2.
Data is then saved to .csv.

Also extract mean values of features and the mean effort response.
'''
import Utils as util
import pandas as pd
import pickle

feat_to_select = [0, 1, 2, 3, 4, 6, 7] # candidate features we consider 
database_path = "W:/staff-umbrella/perfectfit/Exp0/2021_04_01_Pilot_chatbot.db"

# People who gave clearly nonsensical planning/reflection answers in session 2 or 1
# We do not want to include their data
session2_nonsens_answers_p_ids = pd.read_csv("W:/staff-umbrella/perfectfit/Exp0/Extract_Data/session2_nonsensical_answers.csv")["Prolific_ID"].tolist()
session1_nonsens_answers_p_ids = pd.read_csv("W:/staff-umbrella/perfectfit/Exp0/Extract_Data/session1_nonsensical_answers.csv")["Prolific_ID"].tolist()
excluded_ids = [session1_nonsens_answers_p_ids, session2_nonsens_answers_p_ids]

data, feat_means, user_ids, effort_mean = util.gather_data_post_sess_2(database_path, 
                                                                       feat_to_select,
                                                                       excluded_ids)


df = pd.DataFrame(data, columns = ['s0', 's1', 'a', 'r'])
df.to_csv('data_samples_post_sess_2.csv', index = False)

# feature means of candidate features considering sessions 1 and 2
with open('Post_Sess_2_Feat_Means', 'wb') as f:
    pickle.dump(feat_means, f)

# Mean value of effort responses from session 2
with open("Post_Sess_2_Effort_Mean", "wb") as f:
    pickle.dump(effort_mean, f)
   
# user IDs -> needed for group 4 to compute their individual policies
with open('IDs', 'wb') as f:
    pickle.dump(user_ids, f)
