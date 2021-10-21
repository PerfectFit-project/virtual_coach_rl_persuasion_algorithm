"""
Extract demographics etc. data from Prolific pre-screening and from
Qualtrics pre-questionnaire.
Demographics that are extracted are gender identity, personality and TTM-phase
for becoming more physically active.
Also determines which people passed the pre-questionnaire.
We need the demographics data for the group assignments and for algorithm
level 4.
"""


import pandas as pd


# Min. allowed time for pre-questionnaire in seconds
MIN_TIME_PRE_QUESTIONNAIRE = 101.706


# Load Prolific pre-screening export.
# TODO: replace title in the end
df_prol = pd.read_csv("2021_06_09_Final_Pre-screening_incl_batch_6_prolific_export_60ac97cd97f5cd7420b82a60.csv")
# Remove rows of people who returned their submission or whose submission was rejected.
# Also remove people who timed out in the Pre-screening
df_prol = df_prol[df_prol['status'] != 'RETURNED']
df_prol = df_prol[df_prol['status'] != 'REJECTED']
df_prol = df_prol[df_prol['status'] != 'TIMED-OUT']
df_prol = df_prol.reset_index()


# Load Qualtrics export from pre-questionnaire.
# Assumes that data is exported as .csv and values are recorded as numeric ones.
# TODO: replace title in the end for the right questionnaire export
df_q = pd.read_csv("Final+-+Pre-Questionnaire_June+10,+2021_19.00_incl_batch_5_6_all.csv")
# Remove the first 2 rows which do not contain responses
df_q = df_q.iloc[2:]
df_q = df_q.reset_index()
# Remove un-finished responses in the pre-questionnaire
df_q = df_q[df_q['Finished']== "1"]
# Reset index after possibly removing some rows via previous statement
df_q = df_q.reset_index()
# Number of qualtrics responses
num_resp = len(df_q)


# Attention checks in pre-questionnaire
# Q2_3: need to answer 1/2 (i.e. disagree)
# Q7_7: need to answer 1/2 (i.e. disagree)
# Q11_4: need to answer 5/4 (i.e. agree)
# Q11_6: need to answer 1/2 (i.e. disagree)
# Q15_5: need to answer 5 (i.e. agree strongly)
ac = ['Q2_3', 'Q7_7', 'Q11_4', 'Q11_6', 'Q15_5']
ac_true = [["1", "2"], ["1", "2"], ["4", "5"], ["1", "2"], ["5"]]
num_ac = 5 # total number of attention checks in pre-questionnaire
num_ac_failed_ok = 1 # failing this many attention checks is still ok
num_ac_failed = [sum([1 for i in range(num_ac) if not df_q.loc[j, ac[i]] in ac_true[i]]) for j in range(num_resp)]


df_q['num_ac_failed'] = num_ac_failed


# We also look at the completion time of the pre-questionnaire
# So we load the data from Prolific about the pre-questionnaire
# TODO: put the correct path in the end
# TODO: make sure that this export contains the data from all previous versions of the pre-questionnaire
df_time = pd.read_csv("2021_06_10_Final_Pre-questionnaire_prolific_export_60a7c4c6b9484051c59b85d5_MergedAll_1859.csv")
time_taken = [[df_time.loc[j, 'time_taken'] for j in range(len(df_time)) if df_time.loc[j, 'participant_id'] == df_q.loc[i, 'PROLIFIC_PID']][0] for i in range(len(df_q))]
df_q['time_taken_preq'] = time_taken


# Determine who passed/failed enough/too many attention checks and people who were too fast
df_q_passed_ac = df_q[df_q['num_ac_failed'] <= num_ac_failed_ok]
df_q_passed_ac_slow = df_q_passed_ac[df_q_passed_ac['time_taken_preq'] >= MIN_TIME_PRE_QUESTIONNAIRE]


# People who passed the attention checks but were too fast
df_q_too_fast = df_q_passed_ac[df_q_passed_ac["time_taken_preq"] < MIN_TIME_PRE_QUESTIONNAIRE]


# Get prolific ID of the 3 types of people
df_too_fast_pids = df_q_too_fast['PROLIFIC_PID'] # pay them, but don't continue with them
df_q_passed_ac_pids = df_q_passed_ac_slow['PROLIFIC_PID'] # pay them and continue with them
df_q_failed_ac_pids = df_q[df_q['num_ac_failed'] > num_ac_failed_ok]['PROLIFIC_PID'] # don't pay them


# Personality and PA-TTM answers
df_pers_TTM = df_q_passed_ac[['PROLIFIC_PID', 'Q1_1', 'Q1_2', 'Q1_3', 'Q1_4', 'Q1_5',
                              'Q1_6', 'Q1_7', 'Q1_8', 'Q1_9', 'Q1_10', 'Q6']]
# Reset index. Otherwise, some indices might not exist if people failed too many attention checks.
# And then the code to extract the genders fails.
df_pers_TTM = df_pers_TTM.reset_index()


# Participant genders
gender = [[df_prol.loc[j, 'Gender identity'] for j in range(len(df_prol)) if df_prol.loc[j, 'participant_id'] == df_pers_TTM.loc[i, 'PROLIFIC_PID']] for i in range(len(df_pers_TTM))]
df_pers_TTM['Gender identity'] = [i[0] for i in gender]
# There might be some missing gender identities.
df_pers_TTM['Gender identity'] = df_pers_TTM['Gender identity'].fillna('')


# Encode gender to numbers between 0 and 1
# We only distinguish between the 3 categories male/female/everything else
# Trans males do not necessarily identify as male, so we do not group them to male.
gen_to_num = {'Male': 0, 'Trans Male/Trans Man': 0.5, 'Trans Female/Trans Woman': 0.5, 
              'Genderqueer/Gender Non Conforming': 0.5, 'Different Identity': 0.5, 
              'Rather not say': 0.5, 'Female': 1}
# There might be nan values
df_pers_TTM['Gender identity'] = [gen_to_num[i] if not i == "" else 0.5 for i in df_pers_TTM['Gender identity'].tolist()]


# Encode PA-TTM stage in number and rename corresponding column to PA-TTM
df_pers_TTM['Q6'] = pd.to_numeric(df_pers_TTM["Q6"])
df_pers_TTM = df_pers_TTM.rename(columns = {'Q6': "PA-TTM"})


# Scale PA-TTM stage to [0, 1] (originally, values are from 1 to 5)
df_pers_TTM['PA-TTM'] = (df_pers_TTM['PA-TTM'] - 1)/4.0


# Compute personality
df_pers_TTM['Extraversion'] = [0.5 * (int(df_pers_TTM.loc[i, 'Q1_1']) - (int(df_pers_TTM.loc[i, 'Q1_6']) - 8)) for i in range(len(df_pers_TTM))]
df_pers_TTM['Agreeableness'] = [0.5 * (int(df_pers_TTM.loc[i, 'Q1_7']) - (int(df_pers_TTM.loc[i, 'Q1_2']) - 8)) for i in range(len(df_pers_TTM))]
df_pers_TTM['Conscientiousness'] = [0.5 * (int(df_pers_TTM.loc[i, 'Q1_3']) - (int(df_pers_TTM.loc[i, 'Q1_8']) - 8)) for i in range(len(df_pers_TTM))]
df_pers_TTM['ES'] = [0.5 * (int(df_pers_TTM.loc[i, 'Q1_9']) - (int(df_pers_TTM.loc[i, 'Q1_4']) - 8)) for i in range(len(df_pers_TTM))]
df_pers_TTM['OE'] = [0.5 * (int(df_pers_TTM.loc[i, 'Q1_5']) - (int(df_pers_TTM.loc[i, 'Q1_10']) - 8)) for i in range(len(df_pers_TTM))]


# Drop columns with original input for personality
df_pers_TTM = df_pers_TTM.drop(columns = ['Q1_1', 'Q1_2', 'Q1_3', 'Q1_4', 'Q1_5',
                              'Q1_6', 'Q1_7', 'Q1_8', 'Q1_9', 'Q1_10'])


# Scale personality columns to [0, 1] (originally, values are between 1 and 7)
for col in ['Extraversion', 'Agreeableness', 'Conscientiousness', 'ES', 'OE']:
    df_pers_TTM[col] = (df_pers_TTM[col] - 1)/6.0


# Save prolific IDs of people who passed/failed/were too fast for the pre-questionnaire
df_q_passed_ac_pids.to_csv("pre-questionnaire_passed_p_ids.csv", index = False)
df_q_failed_ac_pids.to_csv("pre-questionnaire_failed_p_ids.csv", index = False)
df_too_fast_pids.to_csv("pre-questionnaire_too_fast_p_ids.csv", index = False)


# Save gender, PA-TTM and personality data
df_pers_TTM.to_csv('pers_PA-TTM_gender.csv', index = False)
