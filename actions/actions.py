# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions

from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from rasa_sdk.events import ConversationResumed
from rasa_sdk.events import UserUtteranceReverted
from rasa_sdk.events import FollowupAction
from rasa_sdk.forms import FormAction
import pickle
import sqlite3
import pandas as pd
import random
import numpy as np
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib, ssl
from string import Template

# Activities
df_act = pd.read_excel("Activities.xlsx")
df_act['Exclusion'] = df_act['Exclusion'].str.strip('()').str.split(',')
for row in df_act.loc[df_act['Exclusion'].isnull(), 'Exclusion'].index:
    df_act.at[row, 'Exclusion'] = []
df_act['Prerequisite'] = df_act['Prerequisite'].str.strip('()').str.split(',')
for row in df_act.loc[df_act['Prerequisite'].isnull(), 'Prerequisite'].index:
    df_act.at[row, 'Prerequisite'] = []
num_act = len(df_act)
# activity indices per category
s_ind = [i for i in range(len(df_act)) if df_act.loc[i, 'Category'][0] == 'S']
pa_ind = [i for i in range(len(df_act)) if df_act.loc[i, 'Category'][0] == 'P']

# Persuasive Messages
df_mess = pd.read_csv("all_messages.csv")
num_mess_per_type = [6, 4, 4, 3]
NUM_PERS_TYPES = 4

# Moods, sorted by quadrant w.r.t. valence and arousal
moods_ha_lv = ["afraid", "alarmed", "annoyed", "distressed", "angry", 
               "frustrated"]
moods_la_lv = ["miserable", "depressed", "gloomy", "tense", "droopy", "sad", 
               "tired", "bored", "sleepy"] # sleepy actually in different quadrant
moods_la_hv = ["content", "serene", "calm", "relaxed", "tranquil"]
moods_ha_hv = ["satisfied", "pleased", "delighted", "happy", "glad", 
               "astonished", "aroused", "excited"]

# function to extract custom data from rasa webchat (in my case only the prolific id)
def extract_metadata_from_tracker(tracker: Tracker):
    events = tracker.current_state()['events']
    user_events = []
    for e in events:
        if e['event'] == 'user':
            user_events.append(e)

    return user_events[-1]['metadata']

# answer based on mood
class ActionAnswerMood(Action):
    def name(self):
        return "action_answer_mood"

    async def run(self, dispatcher, tracker, domain):
        
        curr_mood = tracker.get_slot('mood') #tracker.latest_message['entities'][0]['value']
        
        if curr_mood == "neutral":
            dispatcher.utter_message(template="utter_mood_neutral")
        elif curr_mood in moods_ha_lv:
            dispatcher.utter_message(template="utter_mood_negative_valence_high_arousal_quadrant")
        elif curr_mood in moods_la_lv:
            dispatcher.utter_message(template="utter_mood_negative_valence_low_arousal_quadrant")
        elif curr_mood in moods_la_hv:
            dispatcher.utter_message(template="utter_mood_positive_valence_low_arousal_quadrant")
        else:
            dispatcher.utter_message(template="utter_mood_positive_valence_high_arousal_quadrant")
        
        return []
    
# Choose an activity for the user
class ActionChooseActivity(Action):
    def name(self):
        return "action_choose_activity"

    async def run(self, dispatcher, tracker, domain):
        
        # reset random seed
        random.seed(datetime.now())
        
        curr_act_ind_list = tracker.get_slot('activity_index_list')
        
        if curr_act_ind_list is None:
            curr_act_ind_list = []
        
        # Count how many smoking and PA activities have been done and track excluded activities
        num_s = 0
        num_pa = 0
        excluded = []
        for i in curr_act_ind_list:
            if i in s_ind:
                num_s += 1
            else:
                num_pa += 1
            excluded += df_act.loc[i, 'Exclusion']
            
        # get eligible activities (not done before and not excluded)
        remaining_indices = [ i for i in range(num_act) if not i in curr_act_ind_list and not str(i) in excluded]
            
        # Check if prerequisites for remaining activities are met
        for i in remaining_indices:
            preq = [j for j in df_act.loc[i, 'Prerequisite'] if not str(j) in curr_act_ind_list]
            if len(preq) > 0:
                excluded.append(i)
            
        # get activities that also meet the prerequisites
        remaining_indices = [i for i in remaining_indices if not str(i) in excluded]
        
        if num_s == num_pa:
            # Choose randomly whether to do a smoking or a PA activity
            type_choice = random.choice([0, 1])
            
            # Choose activity from chosen type
            if type_choice == 0:
                # Choose a PA activity
                act_index = random.choice([i for i in remaining_indices if i in pa_ind])
            else:
                # Choose a smoking activity
                act_index = random.choice([i for i in remaining_indices if i in s_ind])
        elif num_s > num_pa:
            # Choose a PA activity
            act_index = random.choice([i for i in remaining_indices if i in pa_ind])
        else:
            # Choose a smoking activity
            act_index = random.choice([i for i in remaining_indices if i in s_ind])
            
        curr_act_ind_list.append(act_index)
        
        return [SlotSet("activity_formulation", df_act.loc[act_index, 'Formulation']), 
                SlotSet("activity_index_list", curr_act_ind_list),
                SlotSet("activity_verb", df_act.loc[act_index, "VerbYou"])]

# Set slot about whether the user completed the assigned activity    
class ActionSetSlotReward(FormAction):

    def name(self):
        return 'action_set_slot_reward'

    async def run(self, dispatcher, tracker, domain):

        reward = tracker.get_slot('reward')
        success = True
        if reward == "0":
            success = False
        
        return [SlotSet("action_success", success)]
    
class ActionGetFreetextActivityComp(FormAction):

    def name(self):
        return 'action_freetext_activity_comp'

    async def run(self, dispatcher, tracker, domain):

        activity_experience = tracker.latest_message['text']
        
        return [SlotSet("activity_experience", activity_experience)]
    
# Read free text reponse for modifications to response for activity experience
class ActionGetFreetextActivityMod(FormAction):

    def name(self):
        return 'action_freetext_activity_mod'

    async def run(self, dispatcher, tracker, domain):

        activity_experience_mod = tracker.latest_message['text']
        
        return [SlotSet("activity_experience_mod", activity_experience_mod)]
    
# Read free text response for user's implementation intention
class ActionGetFreetext(FormAction):

    def name(self):
        return 'action_freetext'

    async def run(self, dispatcher, tracker, domain):

        user_plan = tracker.latest_message['text']
        
        plan_correct = True
        # check syntax
        if not ("if" in user_plan.lower()):
            plan_correct = False
        elif len(user_plan) <= 6:
            plan_correct = False
        else:
            dispatcher.utter_message(template="utter_thank_you_planning")
        
        return [SlotSet("action_planning_answer", user_plan),
                SlotSet("plan_correct", plan_correct)]
    
# Read free text response for user's satifaction
class ActionGetSatisfaction(FormAction):

    def name(self):
        return 'action_get_satisfaction'

    async def run(self, dispatcher, tracker, domain):

        satis = tracker.latest_message['text']
        
        # remove white spaces
        satis = "".join(satis.split())
        
        correct = True
        # check syntax
        try:
            satis_float = float(satis)
            if satis_float > 10:
                correct = False
            elif satis_float < -10:
                correct = False
        except ValueError:
            correct = False # cannot be cast to float
        
        return [SlotSet("user_satisfaction", satis),
                SlotSet("satisf_correct", correct)]
    
# Sets slots for later sessions
class ActionSetSession(Action):
    def name(self) -> Text:
        return "action_set_session"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        metadata = extract_metadata_from_tracker(tracker)
        user_id = metadata['userid']
        print("ID:", user_id)
        
        # create db connection
        try:
            #sqlite_connection = sqlite3.connect('chatbot.db') # TODO
            sqlite_connection = sqlite3.connect('db_scripts/chatbot.db')
            cursor = sqlite_connection.cursor()
            print("Connection created for user ", user_id)
            sqlite_select_query = """SELECT * from users WHERE id = ?"""
            cursor.execute(sqlite_select_query, (user_id,))
            data = cursor.fetchall()
            cursor.close()

        except sqlite3.Error as error:
            print("Error while connecting to sqlite", error)
        finally:
            if (sqlite_connection):
                sqlite_connection.close()
                print("Connection closed for user ", user_id)

        try:
            # load data from previous sessions about activities and actions
            activity_index_list = [int(i) for i in data[0][18].split('|')]
            activity_verb_prev = df_act.loc[activity_index_list[-1], "VerbYou"]
            return [SlotSet("activity_index_list", activity_index_list), 
                    SlotSet("action_index_list", [int (i) for i in data[0][19].split('|')]),
                    SlotSet("activity_verb_prev", activity_verb_prev),
                    SlotSet("action_type_index_list", [int (i) for i in data[0][25].split('|')])]
           
        except NameError:
            dispatcher.utter_message("Something went wrong, please close this session and contact researcher (n.albers@tudelft.nl).")
    
class ActionSendEmail(Action):
    def name(self):
        return "action_send_email"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
    
        metadata = extract_metadata_from_tracker(tracker)
        user_id = metadata['userid'] # Anurag: '5f970a74069a250711aaa695'
        
        ssl_port = 465
        with open('x.txt', 'r') as f:
            x = f.read()
        smtp = "smtp.web.de" # for web.de: smtp.web.de
        email = "nele.96@web.de"
        user_email = user_id + "@email.prolific.co"
        user_email = "n.albers@tudelft.nl"
        
        with open('reminder_template.txt', 'r', encoding='utf-8') as template_file:
            message_template = Template(template_file.read())
        context = ssl.create_default_context()
        
        persuasion = tracker.get_slot('message_formulation')
        activity = tracker.get_slot('activity_formulation')
    
        # set up the SMTP server
        with smtplib.SMTP_SSL(smtp, ssl_port, context = context) as server:
            server.login(email, x)
        
            msg = MIMEMultipart() # create a message
            
            # add in the actual person name to the message template
            message = message_template.substitute(PERSON_NAME ="Study Participant",
                                                  ACTIVITY= activity,
                                                  PERSUASION = persuasion)
        
            # setup the parameters of the message
            msg['From'] = email
            msg['To']=  user_email
            msg['Subject'] = "Prolific Reminder - Computerized Health Coaching"
            
            # add in the message body
            msg.attach(MIMEText(message, 'plain'))
            
            # send the message via the server set up earlier.
            server.send_message(msg)
            
            del msg
            
class ActionSendEmailLast(Action):
    def name(self):
        return "action_send_email_last"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
    
        metadata = extract_metadata_from_tracker(tracker)
        user_id = metadata['userid'] # Anurag: '5f970a74069a250711aaa695'
        
        ssl_port = 465
        with open('x.txt', 'r') as f:
            x = f.read()
        smtp = "smtp.web.de" # for web.de: smtp.web.de
        email = "nele.96@web.de"
        user_email = user_id + "@email.prolific.co"
        user_email = "n.albers@tudelft.nl"
        
        with open('reminder_template_last_session.txt', 'r', encoding='utf-8') as template_file:
            message_template = Template(template_file.read())
        context = ssl.create_default_context()
        
        activity = tracker.get_slot('activity_formulation')
    
        # set up the SMTP server
        with smtplib.SMTP_SSL(smtp, ssl_port, context = context) as server:
            server.login(email, x)
        
            msg = MIMEMultipart() # create a message
            
            # add in the actual person name to the message template
            message = message_template.substitute(PERSON_NAME="Study Participant",
                                                  ACTIVITY= activity)
        
            # setup the parameters of the message
            msg['From'] = email
            msg['To']=  user_email
            msg['Subject'] = "Prolific Reminder - Computerized Health Coaching"
            
            # add in the message body
            msg.attach(MIMEText(message, 'plain'))
            
            # send the message via the server set up earlier.
            server.send_message(msg)
            
            del msg

class ActionSaveSession(Action):
    def name(self):
        return "action_save_session"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        metadata = extract_metadata_from_tracker(tracker)
        user_id = metadata['userid']
        
        # Load slot values
        mood = tracker.get_slot('mood')
        attention_check = tracker.get_slot('attention_check')
        attention_check_2 = tracker.get_slot('attention_check_2')
        activity_index_list  = '|'.join([str(i) for i in tracker.get_slot('activity_index_list')])
        action_type_index_list  = '|'.join([str(i) for i in tracker.get_slot('action_type_index_list')])
        action_index_list = '|'.join([str(i) for i in tracker.get_slot('action_index_list')])
        state = '|'.join([tracker.get_slot('state_1'), tracker.get_slot('state_2'), 
                          tracker.get_slot('state_3'), tracker.get_slot('state_4'),
                          tracker.get_slot('state_5'), tracker.get_slot('state_6'),
                          tracker.get_slot('state_7'), tracker.get_slot('state_8'),
                          tracker.get_slot('state_9'), tracker.get_slot('state_10')])
        
        # create db connection
        try:
            sqliteConnection = sqlite3.connect('db_scripts/chatbot.db')
            #sqliteConnection = sqlite3.connect('chatbot.db') # TODO
            cursor = sqliteConnection.cursor()
            print("Successfully connected to SQLite")
            sqlite_select_query = """SELECT * from users WHERE id = ?"""
            cursor.execute(sqlite_select_query, (user_id,))
            data = cursor.fetchall()
            print(data)
            sessions_done = 0;
            
            if not data:
                sessions_done = 1
                action_planning_answer = tracker.get_slot('action_planning_answer')
                data_tuple = (user_id, sessions_done, mood, action_planning_answer, 
                              attention_check, attention_check_2, activity_index_list,
                              action_index_list, state, action_type_index_list)
                sqlite_query = """INSERT INTO users (id, sessions_done, mood_list, action_planning_answer0, attention_check_list, attention_check_2_list, activity_index_list, action_index_list, state_0, action_type_index_list) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
                dispatcher.utter_message(template="utter_goodbye_not_last")
                link = "https://tudelft.eu.qualtrics.com/jfe/form/SV_3VmOMw3USprKQv3?PROLIFIC_PID=" + str(user_id) + "&Group=2"
                dispatcher.utter_message(link)
                
            elif data[0][1] == 1:
                sessions_done = 2
                action_planning_answer = tracker.get_slot('action_planning_answer')
                mood_list = '|'.join([data[0][2], mood])
                attention_check_list = data[0][16].split('|')
                attention_check_list.append(attention_check)
                attention_check_list = '|'.join(attention_check_list)
                attention_check_2_list = data[0][17].split('|')
                attention_check_2_list.append(attention_check_2)
                attention_check_2_list = '|'.join(attention_check_2_list)
                activity_experience = tracker.get_slot('activity_experience')
                activity_experience_mod = tracker.get_slot('activity_experience_mod')
                reward = tracker.get_slot('reward')
                data_tuple = (sessions_done, mood_list, action_planning_answer, 
                              attention_check_list, attention_check_2_list, activity_index_list,
                              action_index_list, state, activity_experience, 
                              activity_experience_mod, reward, action_type_index_list, user_id)
                print("Tuple session 2:", data_tuple)
                sqlite_query = """UPDATE users SET sessions_done = ?, mood_list = ?, action_planning_answer1 = ?, attention_check_list = ?, attention_check_2_list = ?, activity_index_list = ?, action_index_list = ?, state_1 = ?, activity_experience1 = ?, activity_experience_mod1 = ?, reward_list = ?, action_type_index_list = ? WHERE id = ?"""
                dispatcher.utter_message(template="utter_goodbye_not_last")
                link = "https://tudelft.eu.qualtrics.com/jfe/form/SV_ebsYp1kHo3yrzFj?PROLIFIC_PID=" + str(user_id) + "&Group=2"
                dispatcher.utter_message(link)
                
            elif data[0][1] == 2:
                sessions_done = 3
                action_planning_answer = tracker.get_slot('action_planning_answer')
                satisf = tracker.get_slot('user_satisfaction')
                group = int(tracker.get_slot('study_group'))
                mood_list = data[0][2].split('|')
                mood_list.append(mood)
                mood_list = '|'.join(mood_list)
                attention_check_list = data[0][16].split('|')
                attention_check_list.append(attention_check)
                attention_check_list = '|'.join(attention_check_list)
                attention_check_2_list = data[0][17].split('|')
                attention_check_2_list.append(attention_check_2)
                attention_check_2_list = '|'.join(attention_check_2_list)
                activity_experience = tracker.get_slot('activity_experience')
                activity_experience_mod = tracker.get_slot('activity_experience_mod')
                reward_list = '|'.join([data[0][7], tracker.get_slot('reward')])
                data_tuple = (sessions_done, mood_list, action_planning_answer, 
                              attention_check_list, attention_check_2_list, activity_index_list,
                              action_index_list, state, activity_experience, 
                              activity_experience_mod, reward_list, 
                              action_type_index_list, group, satisf, user_id)
                print("Tuple session 3:", data_tuple)
                sqlite_query = """UPDATE users SET sessions_done = ?, mood_list = ?, action_planning_answer2 = ?, attention_check_list = ?, attention_check_2_list = ?, activity_index_list = ?, action_index_list = ?, state_2 = ?, activity_experience2 = ?, activity_experience_mod2 = ?, reward_list = ?, action_type_index_list = ?, study_group = ?, user_satisfaction2 = ? WHERE id = ?"""
                dispatcher.utter_message(template="utter_goodbye_not_last")
                link = "https://tudelft.eu.qualtrics.com/jfe/form/SV_ebsYp1kHo3yrzFj?PROLIFIC_PID=" + str(user_id) + "&Group=2"
                dispatcher.utter_message(link)   
            
            elif data[0][1] == 3:
                sessions_done = 4
                action_planning_answer = tracker.get_slot('action_planning_answer')
                mood_list = data[0][2].split('|')
                mood_list.append(mood)
                mood_list = '|'.join(mood_list)
                attention_check_list = data[0][16].split('|')
                attention_check_list.append(attention_check)
                attention_check_list = '|'.join(attention_check_list)
                attention_check_2_list = data[0][17].split('|')
                attention_check_2_list.append(attention_check_2)
                attention_check_2_list = '|'.join(attention_check_2_list)
                activity_experience = tracker.get_slot('activity_experience')
                activity_experience_mod = tracker.get_slot('activity_experience_mod')
                reward_list = data[0][7].split('|')
                reward_list.append(tracker.get_slot('reward'))
                reward_list = '|'.join(reward_list)
                data_tuple = (sessions_done, mood_list, action_planning_answer, 
                              attention_check_list, attention_check_2_list, activity_index_list,
                              action_index_list, state, activity_experience, 
                              activity_experience_mod, reward_list, 
                              action_type_index_list, user_id)
                print("Tuple session 4:", data_tuple)
                sqlite_query = """UPDATE users SET sessions_done = ?, mood_list = ?, action_planning_answer3 = ?, attention_check_list = ?, attention_check_2_list = ?, activity_index_list = ?, action_index_list = ?, state_3 = ?, activity_experience3 = ?, activity_experience_mod3 = ?, reward_list = ?, action_type_index_list = ? WHERE id = ?"""
                dispatcher.utter_message(template="utter_goodbye_not_last")
                link = "https://tudelft.eu.qualtrics.com/jfe/form/SV_ebsYp1kHo3yrzFj?PROLIFIC_PID=" + str(user_id) + "&Group=2"
                dispatcher.utter_message(link)   
                
            elif data[0][1] == 4:
                sessions_done = 5
                satisf = tracker.get_slot('user_satisfaction')
                mood_list = data[0][2].split('|')
                mood_list.append(mood)
                mood_list = '|'.join(mood_list)
                attention_check_list = data[0][16].split('|')
                attention_check_list.append(attention_check)
                attention_check_list = '|'.join(attention_check_list)
                attention_check_2_list = data[0][17].split('|')
                attention_check_2_list.append(attention_check_2)
                attention_check_2_list = '|'.join(attention_check_2_list)
                activity_experience = tracker.get_slot('activity_experience')
                activity_experience_mod = tracker.get_slot('activity_experience_mod')
                reward_list = data[0][7].split('|')
                reward_list.append(tracker.get_slot('reward'))
                reward_list = '|'.join(reward_list)
                data_tuple = (sessions_done, mood_list,
                              attention_check_list, attention_check_2_list, activity_index_list,
                              action_index_list, state, activity_experience, 
                              activity_experience_mod, reward_list, 
                              action_type_index_list, satisf, user_id)
                print("Tuple session 4:", data_tuple)
                sqlite_query = """UPDATE users SET sessions_done = ?, mood_list = ?, attention_check_list = ?, attention_check_2_list = ?, activity_index_list = ?, action_index_list = ?, state_4 = ?, activity_experience4 = ?, activity_experience_mod4 = ?, reward_list = ?, action_type_index_list = ?, user_satisfaction4 = ? WHERE id = ?"""
                dispatcher.utter_message(template="utter_goodbye_not_last")
                link = "https://tudelft.eu.qualtrics.com/jfe/form/SV_ebsYp1kHo3yrzFj?PROLIFIC_PID=" + str(user_id) + "&Group=2"
                dispatcher.utter_message(link)    
            #elif data[0][1] == 5:
             #   print("success")
            else:
                dispatcher.utter_message("Something went wrong, please contact researcher: n.albers@tudelft.nl.")
            
            cursor.execute(sqlite_query, data_tuple)
            sqliteConnection.commit()
            cursor.close()

        except sqlite3.Error as error:
            print("Error while connecting to sqlite", error)
        finally:
            if (sqliteConnection):
                sqliteConnection.close()
                print("The SQLite connection is closed")
        # connection closed
        return []
    
class ActionGetGroup(Action):

    def name(self):
        return 'action_get_group'

    async def run(self, dispatcher, tracker, domain):
        
        # Group assignment # TODO
        df_group_ass = pd.read_csv("assignment.csv", dtype={'ID':'string'})

        # get user ID
        metadata = extract_metadata_from_tracker(tracker)
        user_id = metadata['userid']
        
        # get pre-computed group
        group = str(df_group_ass[df_group_ass['ID'] == user_id]["Group"].tolist()[0])
        
        print(group)
        
        return [SlotSet("study_group", group)]
    
# Return best (based on group) or random persuasion
class ActionChoosePersuasion(Action):
    def name(self):
        return "action_choose_persuasion"

    async def run(self, dispatcher, tracker, domain):
        
        # reset random seed
        random.seed(datetime.now())
        
        # load slots
        curr_act_ind_list = tracker.get_slot('activity_index_list')
        curr_action_ind_list = tracker.get_slot('action_index_list')
        curr_action_type_ind_list = tracker.get_slot('action_type_index_list')
        curr_activity = curr_act_ind_list[-1]
        
        # study group
        group = int(tracker.get_slot('study_group'))
        
        if group == 0:
            
            print("Persuasion level 1")
        
            # Load pre-computed list with best actions
            with open('Post_Sess_2/Level_1_Optimal_Policy', 'rb') as f:
                p = pickle.load(f)
            
            # Select a persuasion type randomly (there could be multiple best ones)
            pers_type = random.choice(p)
            
        elif group == 1:
            
            print("Persuasion level 2")
            
            with open('Post_Sess_2/Level_2_Optimal_Policy', 'rb') as f:
                p = pickle.load(f)
                
            with open('Post_Sess_2/Level_2_G_algorithm_chosen_features', 'rb') as f:
                feat = pickle.load(f)
            
            # Load mean values of features based on first 2 sessions
            with open('Post_Sess_2/Post_Sess_2_Feat_Means', 'rb') as f:
                feat_means = pickle.load(f)
            
            state = [int(tracker.get_slot('state_1')), int(tracker.get_slot('state_2')), 
                     int(tracker.get_slot('state_3')), int(tracker.get_slot('state_4')),
                     int(tracker.get_slot('state_5')), int(tracker.get_slot('state_6')),
                     int(tracker.get_slot('state_7')), int(tracker.get_slot('state_8')),
                     int(tracker.get_slot('state_9')), int(tracker.get_slot('state_10'))]
            
            state = [1 if state[i] >= feat_means[i] else 0 for i in range(7)] # make binary
            state = np.take(np.array(state), feat) # take only selected 3 features
            
            # Sample randomly from best persuasion types in state
            pers_type = random.choice(p[state[0]][state[1]][state[2]])
            
        elif group == 2:
            
            print("Persuasion level 3")
            
            with open('Post_Sess_2/Level_3_Optimal_Policy', 'rb') as f:
                p = pickle.load(f)
            
            with open('Post_Sess_2/Level_3_G_algorithm_chosen_features', 'rb') as f:
                feat = pickle.load(f)
                
            with open('Post_Sess_2/Post_Sess_2_Feat_Means', 'rb') as f:
                feat_means = pickle.load(f)
            
            state = [int(tracker.get_slot('state_1')), int(tracker.get_slot('state_2')), 
                     int(tracker.get_slot('state_3')), int(tracker.get_slot('state_4')),
                     int(tracker.get_slot('state_5')), int(tracker.get_slot('state_6')),
                     int(tracker.get_slot('state_7')), int(tracker.get_slot('state_8')),
                     int(tracker.get_slot('state_9')), int(tracker.get_slot('state_10'))]
            
            state = [1 if state[i] >= feat_means[i] else 0 for i in range(7)] # make binary
            state = np.take(np.array(state), feat) # take only selected 3 features
            
            # Sample randomly from best persuasion types
            pers_type = random.choice(p[state[0]][state[1]][state[2]])
            
        elif group == 3:
            
            print("Persuasion level 4")
             
            # get user ID
            metadata = extract_metadata_from_tracker(tracker)
            user_id = metadata['userid']
            
            with open('Post_Sess_2/Level_4_Optimal_Policy', 'rb') as f:
                p = pickle.load(f)
            
            with open('Post_Sess_2/Level_3_G_algorithm_chosen_features', 'rb') as f:
                feat = pickle.load(f)
                
            with open('Post_Sess_2/Post_Sess_2_Feat_Means', 'rb') as f:
                feat_means = pickle.load(f)
            
            state = [int(tracker.get_slot('state_1')), int(tracker.get_slot('state_2')), 
                     int(tracker.get_slot('state_3')), int(tracker.get_slot('state_4')),
                     int(tracker.get_slot('state_5')), int(tracker.get_slot('state_6')),
                     int(tracker.get_slot('state_7')), int(tracker.get_slot('state_8')),
                     int(tracker.get_slot('state_9')), int(tracker.get_slot('state_10'))]
            
            state = [1 if state[i] >= feat_means[i] else 0 for i in range(7)] # make binary
            state = np.take(np.array(state), feat) # take only selected features
            
            # Sample randomly from best persuasion types
            pers_type = random.choice(p[user_id][state[0]][state[1]][state[2]])
         
        # Sessions 1 and 2: random 
        else:
            
            print("Random persuasion")
            
            if curr_action_ind_list is None:
                curr_action_ind_list = []
            if curr_action_type_ind_list is None:
                curr_action_type_ind_list = []
            
            # Choose persuasion type randomly
            pers_type = random.choice([i for i in range(NUM_PERS_TYPES)])
            
        curr_action_type_ind_list.append(pers_type)
        
        # total number of messages per activity in message dataframe
        num_mess_per_activ = len(df_mess)/len(df_act)
       
        # Determine whether user input is required for persuasion type
        require_input = False
        if pers_type == 3:
            require_input = True
        
        # Choose message randomly among messages selected the lowest number of times 
        # for this persuasion type
        counts = [curr_action_ind_list.count(i) for i in range(sum(num_mess_per_type[0:pers_type]), sum(num_mess_per_type[0:pers_type + 1]))]
        min_messages = [i for i in range(num_mess_per_type[pers_type]) if counts[i] == min(counts)]
        message_ind = random.choice(min_messages) + sum(num_mess_per_type[0:pers_type])
        curr_action_ind_list.append(message_ind)
        
        # Determine message
        message = df_mess.loc[int(curr_activity * num_mess_per_activ + message_ind), 'Message']
        
        return [SlotSet("message_formulation", message), 
                SlotSet("action_index_list", curr_action_ind_list),
                SlotSet("action_type_index_list", curr_action_type_ind_list),
                SlotSet("pers_input", require_input)]