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
class AnswerMood(Action):
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
    
class chooseActivity(Action):
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
    
class choosePersuasionRandom(Action):
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
        
        # Choose message randomly among messages selected the lowest number of times
        counts = [curr_action_ind_list.count(i) for i in range(sum(num_mess_per_type[0:pers_type]), sum(num_mess_per_type[0:pers_type + 1]))]
        min_messages = [i for i in range(num_mess_per_type[pers_type]) if counts[i] == min(counts)]
        message_ind = random.choice(min_messages) + sum(num_mess_per_type[0:pers_type])
        curr_action_ind_list.append(message_ind)
        
        # Determine message
        message = df_mess.loc[int(curr_activity * num_mess_per_activ + message_ind), 'Message']
        
        return [SlotSet("message_formulation", message), 
                SlotSet("action_index_list", curr_action_ind_list)]