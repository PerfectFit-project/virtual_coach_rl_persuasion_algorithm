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
import sqlite3
import pandas as pd
import random
import numpy as np

# Activities
df_act = pd.read_excel("C:/Users/nele2/Documents/PerfectFitt/First Experiment/Activities/Activities.xlsx")
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
df_mess = pd.read_csv("C:/Users/nele2/Documents/PerfectFitt/First Experiment/all_messages.csv")
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
    
class ActionChooseActivity(Action):
    def name(self):
        return "action_choose_activity"

    async def run(self, dispatcher, tracker, domain):
        
        curr_act_ind_list = tracker.get_slot('activity_index_list')
        
        if curr_act_ind_list is None:
            curr_act_ind_list = []
        
        # Count how many smoking and PA activities have been done and excluded activities
        num_s = 0
        num_pa = 0
        excluded = []
        for i in curr_act_ind_list:
            if i in s_ind:
                num_s += 1
            else:
                num_pa += 1
            excluded += df_act.loc[i, 'Exclusion']
            
        remaining_indices = [ i for i in range(num_act) if not i in curr_act_ind_list and not str(i) in excluded]
            
        # Check if prerequisites are met
        for i in remaining_indices:
            preq = [j for j in df_act.loc[i, 'Prerequisite'] if not str(j) in curr_act_ind_list]
            if len(preq) > 0:
                excluded.append(i)
                
        remaining_indices = [i for i in remaining_indices if not str(i) in excluded]
        
        if num_s == num_pa:
            # Choose activity randomly among all possible activities
            act_index = random.choice(remaining_indices)
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
    
class ActionSetSlotReward(FormAction):

    def name(self):
        return 'action_set_slot_reward'

    async def run(self, dispatcher, tracker, domain):

        reward = tracker.get_slot('reward')
        success = True
        if reward == "0":
            success = False
        
        return [SlotSet("action_success", success)]
    
class ActionGetGroup(FormAction):

    def name(self):
        return 'action_get_group'

    async def run(self, dispatcher, tracker, domain):

        activity_experience = "df"
        
        return [SlotSet("activity_experience", activity_experience)]
    
class ActionGetFreetextActivityComp(FormAction):

    def name(self):
        return 'action_freetext_activity_comp'

    async def run(self, dispatcher, tracker, domain):

        activity_experience = tracker.latest_message['text']
        
        #print("Activity experience:", activity_experience)
        #print("Intent caught:", tracker.latest_message['intent'].get('name') )
        
        return [SlotSet("activity_experience", activity_experience)]
    
# Read free text reponse for modifications to response for activity experience
class ActionGetFreetextActivityMod(FormAction):

    def name(self):
        return 'action_freetext_activity_mod'

    async def run(self, dispatcher, tracker, domain):

        activity_experience_mod = tracker.latest_message['text']
        
        return [SlotSet("activity_experience_mod", activity_experience_mod)]
    
class ActionGetFreetext(FormAction):

    def name(self):
        return 'action_freetext'

    async def run(self, dispatcher, tracker, domain):

        user_plan = tracker.latest_message['text']
        
        #print("User plan:", user_plan)
        #print(tracker.latest_message['intent'].get('name') )
        
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
    
class ActionSaveSession(Action):
    def name(self):
        return "action_save_session"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        metadata = extract_metadata_from_tracker(tracker)
        user_id = metadata['userid']
        user_id = '111'
        action0 = tracker.get_slot("mood")
        
        # create db connection
        try:
            sqliteConnection = sqlite3.connect('chatbot.db')
            cursor = sqliteConnection.cursor()
            print("Successfully connected to SQLite")
            sqlite_select_query = """SELECT * from users WHERE id = ?"""
            cursor.execute(sqlite_select_query, (user_id,))
            data = cursor.fetchall()
            sessions_done = 0;
            if not data:
                sessions_done = 1
                data_tuple = (user_id, action0, sessions_done)
                sqlite_query = """INSERT INTO users (id, action0, sessions_done) VALUES (?, ?, ?)"""
                dispatcher.utter_message(template="utter_goodbye_not_last")
                link = "https://tudelft.eu.qualtrics.com/jfe/form/SV_3VmOMw3USprKQv3?PROLIFIC_PID=" + str(user_id) + "&Group=2"
                dispatcher.utter_message(link)
            elif data[0][2] == 1:
                sessions_done = 2
                data_tuple = (action0, sessions_done, user_id)
                sqlite_query = """UPDATE users SET action0 = ?, sessions_done = ? WHERE id = ?"""
                dispatcher.utter_message(template="utter_goodbye_not_last")
                link = "https://tudelft.eu.qualtrics.com/jfe/form/SV_ebsYp1kHo3yrzFj?PROLIFIC_PID=" + str(user_id) + "&Group=2"
                dispatcher.utter_message(link)
            elif data [0][2] == 2:
                sessions_done = 3
                data_tuple = (action0, sessions_done, user_id)
                sqlite_query = """UPDATE users SET action0 = ?, sessions_done = ? WHERE id = ?"""
                dispatcher.utter_message(template="utter_goodbye_not_last")
                link = "https://tudelft.eu.qualtrics.com/jfe/form/SV_5A0x6l1ToGlSHyJ?PROLIFIC_PID=" + str(user_id) + "&Group=2"
                dispatcher.utter_message(link)
            else:
                dispatcher.utter_message("Something went wrong, please contact researcher: n.albers@tudelft.nl.")
            # print("user_id: ", user_id, "dd_type: ", dd_type, "dd_present: ", dd_present,
            #       "comfortable_sharing: ", comfortable_sharing, "last tip: ", last_tip, "sessions_done: ",
            #       sessions_done)
            print(user_id, action0, sessions_done)
            # sqlite_insert_userid = """INSERT INTO users (id, dd_type, dd_present, comfortable_sharing, last_tip, sessions_done) VALUES (?, ?, ?, ?, ?, ?)"""
            # data_tuple = (user_id, dd_type, dd_present, comfortable_sharing, last_tip, sessions_done)
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
                 
class ActionChoosePersuasionRandom(Action):
    def name(self):
        return "action_choose_persuasion_random"

    async def run(self, dispatcher, tracker, domain):
        
        curr_act_ind_list = tracker.get_slot('activity_index_list')
        curr_action_ind_list = tracker.get_slot('action_index_list')
        curr_activity = curr_act_ind_list[-1]
        
        if curr_action_ind_list is None:
            curr_action_ind_list = []
        
        num_mess_per_activ = len(df_mess)/len(df_act)
        
        # Choose persuasion type randomly
        pers_type = random.choice([i for i in range(NUM_PERS_TYPES)])
        
        # to test implementation intention
        #pers_type = 3
        
        # Determine whether user input is required for persuasion type
        require_input = False
        if pers_type == 3:
            require_input = True
        
        # Choose message randomly among messages selected the lowest number of times
        counts = [curr_action_ind_list.count(i) for i in range(sum(num_mess_per_type[0:pers_type]), sum(num_mess_per_type[0:pers_type + 1]))]
        min_messages = [i for i in range(num_mess_per_type[pers_type]) if counts[i] == min(counts)]
        message_ind = random.choice(min_messages) + sum(num_mess_per_type[0:pers_type])
        curr_action_ind_list.append(message_ind)
        
        # Determine message
        message = df_mess.loc[int(curr_activity * num_mess_per_activ + message_ind), 'Message']
        
        return [SlotSet("message_formulation", message), 
                SlotSet("action_index_list", curr_action_ind_list),
                SlotSet("pers_input", require_input)]