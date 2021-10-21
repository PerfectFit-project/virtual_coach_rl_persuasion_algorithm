'''
Script to extract which people passed/failed/had an error in any of the 5 sessions with the conversational
agent.

Also extracts the planning/reflection answers for all people who passed the attention
checks in a session. This is so that we can check for non-sensical answers and 
reject and no further invite such people.
'''
import Utils as util
import pandas as pd

database_path = "W:/staff-umbrella/perfectfit/Exp0/2021_06_28_0814_Final_chatbot.db"

# For sessions 1 through 5 with the conversational agent
for session_num in range(1, 6):

    # get IDs of people who passed/failed/had an error w.r.t. attention checks in this session
    user_ids_passed, user_ids_failed, user_ids_error = util.check_attention_checks_session(database_path,
                                                                                           session_num = session_num)
    
    df_user_ids_passed = pd.DataFrame(user_ids_passed, columns = ['PROLIFIC_PID'])
    df_user_ids_failed = pd.DataFrame(user_ids_failed, columns = ['PROLIFIC_PID'])
    
    # something went wrong in a session, i.e. some but not all attention check data was saved.
    df_user_ids_error = pd.DataFrame(user_ids_error, columns = ['PROLIFIC_PID'])
    
    # Save dataframes to .csv-files
    df_user_ids_passed.to_csv("W:/staff-umbrella/perfectfit/Exp0/session" + str(session_num) + "_passed_p_ids.csv", 
                              index = False)
    df_user_ids_failed.to_csv("W:/staff-umbrella/perfectfit/Exp0/session" + str(session_num) + "_failed_p_ids.csv", 
                              index = False)
    df_user_ids_error.to_csv("W:/staff-umbrella/perfectfit/Exp0/session" + str(session_num) + "_error_p_ids.csv", 
                              index = False)
    
    # Get planning/reflection answers for this session
    user_ids_answers, answers, activities, action_types = util.get_planning_reflection_answers(database_path, 
                                                                                        session_num = session_num)
    
    # Only need to look at the answers of people who passed the attention checks and had no error
    indices = [i for i in range(len(user_ids_answers)) if user_ids_answers[i] in user_ids_passed]
    user_ids_answers = list(map(user_ids_answers.__getitem__, indices))
    answers = list(map(answers.__getitem__, indices))
    activities = list(map(activities.__getitem__, indices))
    action_types = list(map(action_types.__getitem__, indices))
    
    df_answers = pd.DataFrame([user_ids_answers, answers, activities, action_types]).transpose()
    df_answers.columns = ["Prolific_ID", "Answer", "Activity", "Action_Type"]
    
    df_answers.to_csv("W:/staff-umbrella/perfectfit/Exp0/session" + str(session_num) + "_answers_passed_p_ids.csv")