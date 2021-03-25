'''
Script to extract which people passed any of the 5 sessions with the conversational
agent.
'''
import Utils as util
import pandas as pd

database_path = 'c:/users/nele2/CA/db_scripts/chatbot.db'

# For sessions 1 through 5 with the conversational agent
for session_num in range(1, 6):

    user_ids_passed, user_ids_failed, user_ids_error = util.check_attention_checks_session(database_path,
                                                                                           session_num = session_num)
    
    df_user_ids_passed = pd.DataFrame(user_ids_passed, columns = ['PROLIFIC_PID'])
    df_user_ids_failed = pd.DataFrame(user_ids_failed, columns = ['PROLIFIC_PID'])
    # something went wrong in a session, i.e. some but not all data was saved.
    df_user_ids_error = pd.DataFrame(user_ids_error, columns = ['PROLIFIC_PID'])
    
    df_user_ids_passed.to_csv("session" + str(session_num) + "_passed_p_ids.csv", 
                              index = False)
    df_user_ids_failed.to_csv("session" + str(session_num) + "_failed_p_ids.csv", 
                              index = False)
    df_user_ids_error.to_csv("session" + str(session_num) + "_error_p_ids.csv", 
                              index = False)