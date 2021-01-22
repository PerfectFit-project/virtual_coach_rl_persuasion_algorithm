'''
Script to extract data samples (i.e. <s, a, r, s'>) from database after session 2.
Data is then saved to .csv.
'''
import Utils as util
import pandas as pd
import pickle

feat_to_select = [0, 1, 2, 3, 4, 6, 7] # candidate features we consider 
data, feat_means, user_ids = util.gather_data_post_sess_2(feat_to_select)

df = pd.DataFrame(data, columns = ['s0', 's1', 'a', 'r'])
df.to_csv('data_samples_post_sess_2.csv', index = False)

# feature means of candidate features considering sessions 1 and 2
with open('Post_Sess_2_Feat_Means', 'wb') as f:
    pickle.dump(feat_means, f)
   
# user IDs -> needed for group 4 (make sure not to download from server)
with open('IDs', 'wb') as f:
    pickle.dump(user_ids, f)