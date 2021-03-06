# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import numpy as np
import pandas as pd
import pickle
import random
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import smtplib, ssl
import sqlite3
from string import Template
import time
from typing import Any, Text, Dict, List

DATABASE_PATH = 'db_scripts/chatbot.db'

# Activities
df_act = pd.read_csv("Activities.csv")
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

# Reflective questions
df_ref = pd.read_csv("reflective_questions.csv")
ref_dict = {} # reflective question for each message
for m in [0, 1, 2, 3]: # Goal
    ref_dict[m] = 0
for m in [4, 5]: # Identity
    ref_dict[m] = 1
for m in [6, 7, 8, 9]: # Consensus
    ref_dict[m] = 2
for m in [10, 11, 12, 13]: # Authority
    ref_dict[m] = 3
for m in [14, 15, 16, 17, 18]: # Planning
    ref_dict[m] = -1

# Persuasive messages and reminder questions
df_mess = pd.read_csv("all_messages.csv")
df_rem = pd.read_csv("all_reminders.csv")

# number of messages per persuasion type in which we send messages
num_mess_per_type = [6, 4, 4, 5] 
# number of persuasion types in which we send a persuasive message
NUM_PERS_TYPES = len(num_mess_per_type)

# Moods, sorted by quadrant w.r.t. valence and arousal
moods_ha_lv = ["afraid", "alarmed", "annoyed", "distressed", "angry", 
               "frustrated"]
moods_la_lv = ["miserable", "depressed", "gloomy", "tense", "droopy", "sad", 
               "tired", "bored", "sleepy"] # sleepy actually in different quadrant
moods_la_hv = ["content", "serene", "calm", "relaxed", "tranquil"]
moods_ha_hv = ["satisfied", "pleased", "delighted", "happy", "glad", 
               "astonished", "aroused", "excited"]

# Function to extract custom data from rasa webchat (in my case only the prolific id)
def extract_metadata_from_tracker(tracker: Tracker):
    events = tracker.current_state()['events']
    user_events = []
    for e in events:
        if e['event'] == 'user':
            user_events.append(e)

    return user_events[-1]['metadata']

# Answer based on mood
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
    
# Pause for 5 seconds
class ActionPauseFive(Action):
    def name(self):
        return "action_pause_five"

    async def run(self, dispatcher, tracker, domain):
        
        time.sleep(5)
        
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
            preq = [j for j in df_act.loc[i, 'Prerequisite'] if not int('0' + j) in curr_act_ind_list and not len(j) == 0]
            if len(preq) > 0:
                excluded.append(str(i))
            
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
        
        return [SlotSet("activity_formulation", df_act.loc[act_index, 'Formulation Chat']), 
                SlotSet("activity_formulation_email", df_act.loc[act_index, 'Formulation Email']),
                SlotSet("activity_index_list", curr_act_ind_list),
                SlotSet("activity_verb", df_act.loc[act_index, "VerbYouShort"])]
    
# Choose an activity for the user in the last session
# Difference is that the activity formulation needs to be adapted since there is no next session.
class ActionChooseActivityLast(Action):
    def name(self):
        return "action_choose_activity_last"

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
            preq = [j for j in df_act.loc[i, 'Prerequisite'] if not int('0' + j) in curr_act_ind_list and not len(j) == 0]
            if len(preq) > 0:
                excluded.append(str(i))
            
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
        
        activity_formulation_email = df_act.loc[act_index, 'Formulation Email']
        
        # replace anything related to the next session as there is no
        # next session after the last session
        activity_formulation_email = activity_formulation_email.replace(" before the next session,", 
                                                                        "")
        activity_formulation_email = activity_formulation_email.replace(" before the next session",
                                                                        "")
        
        return [SlotSet("activity_formulation", df_act.loc[act_index, 'Formulation Chat']), 
                SlotSet("activity_formulation_email", df_act.loc[act_index, 'Formulation Email']),
                SlotSet("activity_index_list", curr_act_ind_list),
                SlotSet("activity_verb", df_act.loc[act_index, "VerbYou"])]

# Set slot about whether the user completed the assigned activity    
class ActionSetSlotReward(Action):

    def name(self):
        return 'action_set_slot_reward'

    async def run(self, dispatcher, tracker, domain):

        reward = int(tracker.get_slot('reward'))
        success = 1 # answers 4-6 on the scale, i.e. medium effort
        # Scale goes from 0 (nothing) to 10 (extremely strong).
        # For now, we somewhat arbitrarily set the ranges. After the pilot,
        # we might adapt what we consider "success" and what a failure.
        if reward < 4:
            success = 0 # answers 0-3 on scale, i.e. low effort
        elif reward > 6:
            success = 2 # answers 7-10 on scale, i.e. high effort
        
        return [SlotSet("action_success", success)]
    
# Set slot about experience with activity
class ActionGetFreetextActivityComp(Action):

    def name(self):
        return 'action_freetext_activity_comp'

    async def run(self, dispatcher, tracker, domain):

        activity_experience = tracker.latest_message['text']
        
        return [SlotSet("activity_experience", activity_experience)]
    
# Read free text reponse for modifications to response for activity experience
class ActionGetFreetextActivityMod(Action):

    def name(self):
        return 'action_freetext_activity_mod'

    async def run(self, dispatcher, tracker, domain):

        activity_experience_mod = tracker.latest_message['text']
        
        return [SlotSet("activity_experience_mod", activity_experience_mod)]
    
# Read free text response for user's implementation intention/ action planning
class ActionGetFreetext(Action):

    def name(self):
        return 'action_freetext'

    async def run(self, dispatcher, tracker, domain):

        user_plan = tracker.latest_message['text']
        
        plan_correct = True
        # check syntax, i.e. require an "if" and a space afterwards
        if not ("if " in user_plan.lower()):
            plan_correct = False
        # some minimum length is needed
        elif len(user_plan) <= 6:
            plan_correct = False
        
        return [SlotSet("action_planning_answer", user_plan),
                SlotSet("plan_correct", plan_correct)]
    
# Read free text response for user's satifaction.
class ActionGetSatisfaction(Action):

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
			# check bounds
            if satis_float > 10:
                correct = False
            elif satis_float < -10:
                correct = False
        except ValueError:
            correct = False # cannot be cast to float
        
        return [SlotSet("user_satisfaction", satis),
                SlotSet("satisf_correct", correct)]
    
# Read free text response for user's reflection on persuasive message
class ActionGetReflection(Action):

    def name(self):
        return 'action_get_reflection'

    async def run(self, dispatcher, tracker, domain):

        text = tracker.latest_message['text']
        
        return [SlotSet("reflection_answer", text)]
    
# Sets slots for later sessions
class ActionSetSession(Action):
    def name(self) -> Text:
        return "action_set_session"

    async def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        session_loaded = True
        
        # get user ID
        metadata = extract_metadata_from_tracker(tracker)
        user_id = metadata['userid']
        
        # create db connection
        try:
            sqlite_connection = sqlite3.connect(DATABASE_PATH)
            cursor = sqlite_connection.cursor()
            sqlite_select_query = """SELECT * from users WHERE id = ?"""
            cursor.execute(sqlite_select_query, (user_id,))
            data = cursor.fetchall()
            cursor.close()

        except sqlite3.Error as error:
            session_loaded = False
            print("Error while connecting to sqlite", error)
        finally:
            if (sqlite_connection):
                sqlite_connection.close()
        
        activity_verb_prev = ""
        activity_index_list = []
        action_index_list = []
        action_type_index_list = []
        
        try:
            # load data from previous sessions about activities and actions
            activity_index_list = [int(i) for i in data[0][18].split('|')]
            activity_verb_prev = df_act.loc[activity_index_list[-1], "Components"]
            action_index_list = [int (i) for i in data[0][19].split('|')]
            action_type_index_list = [int (i) for i in data[0][25].split('|')]
            
        except NameError:
            session_loaded = False
            print("NameError in action_set_session.")
        
        except IndexError:
            session_loaded = False
            print("IndexError in action_set_session.")
        
        except Exception:
            session_loaded = False
            print("Some exception in action_set_session.")
        
        return [SlotSet("activity_index_list", activity_index_list), 
                SlotSet("action_index_list", action_index_list),
                SlotSet("activity_verb_prev", activity_verb_prev),
                SlotSet("action_type_index_list", action_type_index_list),
                SlotSet("session_loaded", session_loaded)]

# Send reminder email with activity and persuasion after session
class ActionSendEmail(Action):
    def name(self):
        return "action_send_email"

    async def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
    
        # get user ID
        metadata = extract_metadata_from_tracker(tracker)
        user_id = metadata['userid']
        
        ssl_port = 465
        with open('x.txt', 'r') as f:
            x = f.read()
        smtp = "smtp.web.de" # for web.de: smtp.web.de
        with open('email.txt', 'r') as f:
            email = f.read()
        user_email = user_id + "@email.prolific.co"
        
        context = ssl.create_default_context()
    
        # set up the SMTP server
        with smtplib.SMTP_SSL(smtp, ssl_port, context = context) as server:
            server.login(email, x)
        
            msg = MIMEMultipart() # create a message
            
            # our persuasion type is not to send no persuasive message
            if not tracker.get_slot('pers_type_four'):
                with open('reminder_template.txt', 'r', encoding='utf-8') as template_file:
                    message_template = Template(template_file.read())
            
                # add in the actual info to the message template
                message = message_template.substitute(PERSON_NAME ="Study Participant",
                                                      ACTIVITY= tracker.get_slot('activity_formulation_email'),
                                                      PERSUASION = tracker.get_slot('reminder_formulation'))
                
            else:
                with open('reminder_template_noPers.txt', 'r', encoding='utf-8') as template_file:
                    message_template = Template(template_file.read())
            
                # add in the actual info to the message template
                message = message_template.substitute(PERSON_NAME ="Study Participant",
                                                      ACTIVITY= tracker.get_slot('activity_formulation_email'))
        
            # setup the parameters of the message
            msg['From'] = email
            msg['To']=  user_email
            msg['Subject'] = "Activity Reminder - Computerized Health Coaching"
            
            # add in the message body
            msg.attach(MIMEText(message, 'plain'))
            
            # send the message via the server set up earlier.
            server.send_message(msg)
            
            del msg
            
        return []
            
# Send reminder email/message in last session
class ActionSendEmailLast(Action):
    def name(self):
        return "action_send_email_last"

    async def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
    
        metadata = extract_metadata_from_tracker(tracker)
        user_id = metadata['userid']
        
        ssl_port = 465
        with open('x.txt', 'r') as f:
            x = f.read()
        smtp = "smtp.web.de" # for web.de: smtp.web.de
        with open('email.txt', 'r') as f:
            email = f.read()
        user_email = user_id + "@email.prolific.co"
        
        context = ssl.create_default_context()
        
        # set up the SMTP server
        with smtplib.SMTP_SSL(smtp, ssl_port, context = context) as server:
            server.login(email, x)
        
            msg = MIMEMultipart() # create a message
            
            # Our persuasion type is not to send no persuasive message
            if not tracker.get_slot('pers_type_four'):
                with open('reminder_template_last_session.txt', 'r', encoding='utf-8') as template_file:
                    message_template = Template(template_file.read())
            
                # add in the actual information to the message template
                message = message_template.substitute(PERSON_NAME="Study Participant",
                                                  ACTIVITY= tracker.get_slot('activity_formulation_email'),
                                                  PERSUASION = tracker.get_slot('reminder_formulation'))
            else:
                with open('reminder_template_last_session_noPers.txt', 'r', encoding='utf-8') as template_file:
                    message_template = Template(template_file.read())
            
                # add in the actual information to the message template
                message = message_template.substitute(PERSON_NAME="Study Participant",
                                                  ACTIVITY= tracker.get_slot('activity_formulation_email'))
                
            # setup the parameters of the message
            msg['From'] = email
            msg['To']=  user_email
            msg['Subject'] = "Activity Reminder - Computerized Health Coaching"
            
            # add in the message body
            msg.attach(MIMEText(message, 'plain'))
            
            # send the message via the server set up earlier.
            server.send_message(msg)
            
            del msg
            
        return []

class ActionGetGroup(Action):

    def name(self):
        return 'action_get_group'

    async def run(self, dispatcher, tracker, domain):
        
        # Group assignment
        df_group_ass = pd.read_csv("assignment.csv", dtype={'ID':'string'})

        # get user ID
        metadata = extract_metadata_from_tracker(tracker)
        user_id = metadata['userid']
        
        # get pre-computed group
        group = str(df_group_ass[df_group_ass['ID'] == user_id]["Group"].tolist()[0])
        
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
        group = tracker.get_slot('study_group')
        
        # group is not set in sessions 1 and 2
        if len(group) > 0:
            group = int(group)
        
        # We send the best persuasion type overall
        if group == 0:
        
            # Load pre-computed list with best actions
            with open('Post_Sess_2/Level_1_Optimal_Policy', 'rb') as f:
                p = pickle.load(f)
            
            # Select a persuasion type randomly (there could be multiple best ones)
            pers_type = random.choice(p)
           
        # we send the most successful persuasion type in the current user's state
        elif group == 1:
            
            with open('Post_Sess_2/Level_2_Optimal_Policy', 'rb') as f:
                p = pickle.load(f)
                
            with open('Post_Sess_2/Level_2_G_algorithm_chosen_features', 'rb') as f:
                feat = pickle.load(f)
            
            # Load mean values of features based on first 2 sessions
            with open('Post_Sess_2/Post_Sess_2_Feat_Means', 'rb') as f:
                feat_means = pickle.load(f)
            
            state = [int(tracker.get_slot('state_1')), int(tracker.get_slot('state_2')), 
                     int(tracker.get_slot('state_3')), int(tracker.get_slot('state_4')),
                     int(tracker.get_slot('state_5')),
                     int(tracker.get_slot('state_7')), int(tracker.get_slot('state_8'))]
            
            state = [1 if state[i] >= feat_means[i] else 0 for i in range(7)] # make binary
            state = np.take(np.array(state), feat) # take only selected 3 features
            
            # Sample randomly from best persuasion types in state
            pers_type = random.choice(p[state[0]][state[1]][state[2]])
            
        elif group == 2:
            
            with open('Post_Sess_2/Level_3_Optimal_Policy', 'rb') as f:
                p = pickle.load(f)
            
            with open('Post_Sess_2/Level_3_G_algorithm_chosen_features', 'rb') as f:
                feat = pickle.load(f)
                
            with open('Post_Sess_2/Post_Sess_2_Feat_Means', 'rb') as f:
                feat_means = pickle.load(f)
            
            state = [int(tracker.get_slot('state_1')), int(tracker.get_slot('state_2')), 
                     int(tracker.get_slot('state_3')), int(tracker.get_slot('state_4')),
                     int(tracker.get_slot('state_5')),
                     int(tracker.get_slot('state_7')), int(tracker.get_slot('state_8'))]
            
            state = [1 if state[i] >= feat_means[i] else 0 for i in range(7)] # make binary
            state = np.take(np.array(state), feat) # take only selected 3 features
            
            # Sample randomly from best persuasion types
            pers_type = random.choice(p[state[0]][state[1]][state[2]])
            
        elif group == 3:
             
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
                     int(tracker.get_slot('state_5')), 
                     int(tracker.get_slot('state_7')), int(tracker.get_slot('state_8'))]
            
            state = [1 if state[i] >= feat_means[i] else 0 for i in range(7)] # make binary
            state = np.take(np.array(state), feat) # take only selected features
            
            # Sample randomly from best persuasion types
            pers_type = random.choice(p[user_id][state[0]][state[1]][state[2]])
         
        # Random persuasion type (e.g. in session 1 and session 2) 
        else:
            
            if curr_action_ind_list is None:
                curr_action_ind_list = []
            if curr_action_type_ind_list is None:
                curr_action_type_ind_list = []
            
            # Choose persuasion type randomly, including the option
            # not to send a persuasive message
            pers_type = random.choice([i for i in range(NUM_PERS_TYPES + 1)])
            
        curr_action_type_ind_list.append(pers_type)
        
        # total number of messages per activity in message dataframe
        num_mess_per_activ = len(df_mess)/len(df_act)
        
        # Determine whether user input is required for persuasion type
        require_input = False
        if pers_type == 3:
            require_input = True
        
        # Not for persuasion type 4, which is to not send any persuasive message.
        if not pers_type == 4:
            
            pers_type_four = False
        
            # Choose message randomly among messages selected the lowest number of times 
            # for this persuasion type
            counts = [curr_action_ind_list.count(i) for i in range(sum(num_mess_per_type[0:pers_type]), sum(num_mess_per_type[0:pers_type + 1]))]
            min_messages = [i for i in range(num_mess_per_type[pers_type]) if counts[i] == min(counts)]
            message_ind = random.choice(min_messages) + sum(num_mess_per_type[0:pers_type])
            curr_action_ind_list.append(message_ind)
            
            # Determine reflective question (only for persuasion types 0-2)
            ref_type = ref_dict[message_ind]
            ref_question = ""
            if ref_type >= 0:
                # Always pick smoking-related reflective question
                ref_question = df_ref.loc[ref_type, 'QuestionS']
               
            # Determine message and reminder (to be sent in email)
            message = df_mess.loc[int(curr_activity * num_mess_per_activ + message_ind), 'Message']
            reminder = df_rem.loc[int(curr_activity * num_mess_per_activ + message_ind), 'Question']
        
        else:
            message = ""
            reminder = ""
            ref_question = ""
            pers_type_four = True
            curr_action_ind_list.append(-1) # we send no message, so append a -1.
        
        return [SlotSet("message_formulation", message), 
                SlotSet("reminder_formulation", reminder),
                SlotSet("action_index_list", curr_action_ind_list),
                SlotSet("action_type_index_list", curr_action_type_ind_list),
                SlotSet("pers_input", require_input),
                SlotSet("pers_type_four", pers_type_four),
                SlotSet("reflective_question", ref_question)]
    
# Return best (based on group) or random persuasion
# In last session, need to replace parts of messages of the activity planning
# messages, as they talk about the time between this and the next session.
class ActionChoosePersuasionLast(Action):
    def name(self):
        return "action_choose_persuasion_last"

    async def run(self, dispatcher, tracker, domain):
        
        # reset random seed
        random.seed(datetime.now())
        
        # load slots
        curr_act_ind_list = tracker.get_slot('activity_index_list')
        curr_action_ind_list = tracker.get_slot('action_index_list')
        curr_action_type_ind_list = tracker.get_slot('action_type_index_list')
        curr_activity = curr_act_ind_list[-1]
        
        # study group
        group = tracker.get_slot('study_group')
        
        # group is not set in sessions 1 and 2
        if len(group) > 0:
            group = int(group)
        
        if group == 0:
            
            #print("Persuasion level 1")
        
            # Load pre-computed list with best actions
            with open('Post_Sess_2/Level_1_Optimal_Policy', 'rb') as f:
                p = pickle.load(f)
            
            # Select a persuasion type randomly (there could be multiple best ones)
            pers_type = random.choice(p)
            
        elif group == 1:
            
            #print("Persuasion level 2")
            
            with open('Post_Sess_2/Level_2_Optimal_Policy', 'rb') as f:
                p = pickle.load(f)
                
            with open('Post_Sess_2/Level_2_G_algorithm_chosen_features', 'rb') as f:
                feat = pickle.load(f)
            
            # Load mean values of features based on first 2 sessions
            with open('Post_Sess_2/Post_Sess_2_Feat_Means', 'rb') as f:
                feat_means = pickle.load(f)
            
            state = [int(tracker.get_slot('state_1')), int(tracker.get_slot('state_2')), 
                     int(tracker.get_slot('state_3')), int(tracker.get_slot('state_4')),
                     int(tracker.get_slot('state_5')),
                     int(tracker.get_slot('state_7')), int(tracker.get_slot('state_8'))]
            
            state = [1 if state[i] >= feat_means[i] else 0 for i in range(7)] # make binary
            state = np.take(np.array(state), feat) # take only selected 3 features
            
            # Sample randomly from best persuasion types in state
            pers_type = random.choice(p[state[0]][state[1]][state[2]])
            
        
        elif group == 2:
            
            with open('Post_Sess_2/Level_3_Optimal_Policy', 'rb') as f:
                p = pickle.load(f)
            
            with open('Post_Sess_2/Level_3_G_algorithm_chosen_features', 'rb') as f:
                feat = pickle.load(f)
                
            with open('Post_Sess_2/Post_Sess_2_Feat_Means', 'rb') as f:
                feat_means = pickle.load(f)
            
            state = [int(tracker.get_slot('state_1')), int(tracker.get_slot('state_2')), 
                     int(tracker.get_slot('state_3')), int(tracker.get_slot('state_4')),
                     int(tracker.get_slot('state_5')),
                     int(tracker.get_slot('state_7')), int(tracker.get_slot('state_8'))]
            
            state = [1 if state[i] >= feat_means[i] else 0 for i in range(7)] # make binary
            state = np.take(np.array(state), feat) # take only selected 3 features
            
            # Sample randomly from best persuasion types
            pers_type = random.choice(p[state[0]][state[1]][state[2]])
            
        # Persuasion level 4, i.e. highest weighted Q-value
        elif group == 3:
             
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
                     int(tracker.get_slot('state_5')),
                     int(tracker.get_slot('state_7')), int(tracker.get_slot('state_8'))]
            
            state = [1 if state[i] >= feat_means[i] else 0 for i in range(7)] # make binary
            state = np.take(np.array(state), feat) # take only selected features
            
            # Sample randomly from best persuasion types
            pers_type = random.choice(p[user_id][state[0]][state[1]][state[2]])
         
        # Sessions 1 and 2: random  persuasion
        else:
            if curr_action_ind_list is None:
                curr_action_ind_list = []
            if curr_action_type_ind_list is None:
                curr_action_type_ind_list = []
            
            # Choose persuasion type randomly, including the option
            # not to send a persuasive message
            pers_type = random.choice([i for i in range(NUM_PERS_TYPES + 1)])
            
        curr_action_type_ind_list.append(pers_type)
        
        # total number of messages per activity in message dataframe
        num_mess_per_activ = len(df_mess)/len(df_act)
        
        # Determine whether user input is required for persuasion type
        require_input = False
        if pers_type == 3:
            require_input = True
            
        # For persuastion type number 4, we do not send any persuasive message
        if not pers_type == 4:
            
            # our persuastion type is not number 4
            pers_type_four = False
        
            # Choose message randomly among messages selected the lowest number of times 
            # for this persuasion type
            counts = [curr_action_ind_list.count(i) for i in range(sum(num_mess_per_type[0:pers_type]), sum(num_mess_per_type[0:pers_type + 1]))]
            min_messages = [i for i in range(num_mess_per_type[pers_type]) if counts[i] == min(counts)]
            message_ind = random.choice(min_messages) + sum(num_mess_per_type[0:pers_type])
            curr_action_ind_list.append(message_ind)
            
            # Determine reflective question (only for persuasion types 0-2)
            ref_type = ref_dict[message_ind]
            ref_question = ""
            if ref_type >= 0:
                # Always pick smoking-related reflective question
                ref_question = df_ref.loc[ref_type, 'QuestionS']
               
            # Determine message and reminder
            message = df_mess.loc[int(curr_activity * num_mess_per_activ + message_ind), 'Message']
            reminder = df_rem.loc[int(curr_activity * num_mess_per_activ + message_ind), 'Question']
            
            # There is no next session after session 5, so need to adapt action planning messages
            if pers_type == 3:
                message = message.replace("before the next session?", "after this session?")
                message = message.replace("and before the next session", "session")
                reminder = reminder.replace("before the next session?", "after this session?")
        
        # For persuasion type number 4, we do not send any persuasive message
        else:
            message = ""
            reminder = ""
            ref_question = ""
            pers_type_four = True
            curr_action_ind_list.append(-1) # we send no message, so append a -1.
        
        return [SlotSet("message_formulation", message), 
                SlotSet("pers_type_four", pers_type_four),
                SlotSet("reminder_formulation", reminder),
                SlotSet("action_index_list", curr_action_ind_list),
                SlotSet("action_type_index_list", curr_action_type_ind_list),
                SlotSet("pers_input", require_input),
                SlotSet("reflective_question", ref_question)]
    
class ActionSaveSession(Action):
    def name(self):
        return "action_save_session"

    async def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # get user ID
        metadata = extract_metadata_from_tracker(tracker)
        user_id = metadata['userid']
        
        # whether the session has been saved successfully
        session_saved = True
        
        # Load slot values
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
            sqliteConnection = sqlite3.connect(DATABASE_PATH)
            cursor = sqliteConnection.cursor()
            sqlite_select_query = """SELECT * from users WHERE id = ?"""
            cursor.execute(sqlite_select_query, (user_id,))
            data = cursor.fetchall()
            
            sessions_done = 0
            link = ""
            
            # Save data after first session.
            # we still need to save the mood, since we have not saved anything
            # yet in this conversation.
            if not data:
                sessions_done = 1
                mood = tracker.get_slot('mood') # already saved earlier in all other sessions
                action_planning_answer = tracker.get_slot('action_planning_answer')
                reflection_answer = tracker.get_slot('reflection_answer')
                data_tuple = (user_id, sessions_done, mood, action_planning_answer, 
                              attention_check, attention_check_2, activity_index_list,
                              action_index_list, state, action_type_index_list,
                              reflection_answer)
                sqlite_query = """INSERT INTO users (id, sessions_done, mood_list, action_planning_answer0, attention_check_list, attention_check_2_list, activity_index_list, action_index_list, state_0, action_type_index_list, reflection_answer0) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
                link = "https://app.prolific.co/submissions/complete?cc=6C5C40DA"
            
            # save data after second session
            # do not need to save the reward and mood anymore.
            # These have already been saved earlier in the conversation.
            elif data[0][1] == 1:
                sessions_done = 2
                action_planning_answer = tracker.get_slot('action_planning_answer')
                reflection_answer = tracker.get_slot('reflection_answer')
                attention_check_list = data[0][16].split('|')
                attention_check_list.append(attention_check)
                attention_check_list = '|'.join(attention_check_list)
                attention_check_2_list = data[0][17].split('|')
                attention_check_2_list.append(attention_check_2)
                attention_check_2_list = '|'.join(attention_check_2_list)
                activity_experience = tracker.get_slot('activity_experience')
                activity_experience_mod = tracker.get_slot('activity_experience_mod')
                data_tuple = (sessions_done, action_planning_answer, 
                              attention_check_list, attention_check_2_list, 
                              activity_index_list,
                              action_index_list, state, activity_experience, 
                              activity_experience_mod, action_type_index_list, 
                              reflection_answer, user_id)
                
                sqlite_query = """UPDATE users SET sessions_done = ?, action_planning_answer1 = ?, attention_check_list = ?, attention_check_2_list = ?, activity_index_list = ?, action_index_list = ?, state_1 = ?, activity_experience1 = ?, activity_experience_mod1 = ?, action_type_index_list = ?, reflection_answer1 = ? WHERE id = ?"""
                link = "https://app.prolific.co/submissions/complete?cc=5AC5F859"
                
            # do not need to save the reward, mood, and satisfaction anymore.
            # These have already been saved earlier.
            elif data[0][1] == 2:
                sessions_done = 3
                action_planning_answer = tracker.get_slot('action_planning_answer')
                reflection_answer = tracker.get_slot('reflection_answer')
                group = int(tracker.get_slot('study_group'))
                attention_check_list = data[0][16].split('|')
                attention_check_list.append(attention_check)
                attention_check_list = '|'.join(attention_check_list)
                attention_check_2_list = data[0][17].split('|')
                attention_check_2_list.append(attention_check_2)
                attention_check_2_list = '|'.join(attention_check_2_list)
                activity_experience = tracker.get_slot('activity_experience')
                activity_experience_mod = tracker.get_slot('activity_experience_mod')
                data_tuple = (sessions_done, action_planning_answer, 
                              attention_check_list, attention_check_2_list, activity_index_list,
                              action_index_list, state, activity_experience, 
                              activity_experience_mod, 
                              action_type_index_list, group, 
                              reflection_answer, user_id)
                
                sqlite_query = """UPDATE users SET sessions_done = ?, action_planning_answer2 = ?, attention_check_list = ?, attention_check_2_list = ?, activity_index_list = ?, action_index_list = ?, state_2 = ?, activity_experience2 = ?, activity_experience_mod2 = ?, action_type_index_list = ?, study_group = ?, reflection_answer2 = ? WHERE id = ?"""
                link = "https://app.prolific.co/submissions/complete?cc=5C9794E1"
               
            # do not need to save the reward and mood anymore.
            # These have already been saved earlier in the conversation.
            elif data[0][1] == 3:
                sessions_done = 4
                action_planning_answer = tracker.get_slot('action_planning_answer')
                reflection_answer = tracker.get_slot('reflection_answer')
                attention_check_list = data[0][16].split('|')
                attention_check_list.append(attention_check)
                attention_check_list = '|'.join(attention_check_list)
                attention_check_2_list = data[0][17].split('|')
                attention_check_2_list.append(attention_check_2)
                attention_check_2_list = '|'.join(attention_check_2_list)
                activity_experience = tracker.get_slot('activity_experience')
                activity_experience_mod = tracker.get_slot('activity_experience_mod')
                data_tuple = (sessions_done, action_planning_answer, 
                              attention_check_list, attention_check_2_list, activity_index_list,
                              action_index_list, state, activity_experience, 
                              activity_experience_mod,
                              action_type_index_list, 
                              reflection_answer, user_id)
                sqlite_query = """UPDATE users SET sessions_done = ?, action_planning_answer3 = ?, attention_check_list = ?, attention_check_2_list = ?, activity_index_list = ?, action_index_list = ?, state_3 = ?, activity_experience3 = ?, activity_experience_mod3 = ?, action_type_index_list = ?, reflection_answer3 = ? WHERE id = ?"""
                link = "https://app.prolific.co/submissions/complete?cc=4DB200E7"
                
            # save data after last session
            # do not need to save the reward, mood, and satisfaction anymore.
            # These have already been saved earlier.
            elif data[0][1] == 4:
                sessions_done = 5
                action_planning_answer = tracker.get_slot('action_planning_answer')
                reflection_answer = tracker.get_slot('reflection_answer')
                attention_check_list = data[0][16].split('|')
                attention_check_list.append(attention_check)
                attention_check_list = '|'.join(attention_check_list)
                attention_check_2_list = data[0][17].split('|')
                attention_check_2_list.append(attention_check_2)
                attention_check_2_list = '|'.join(attention_check_2_list)
                activity_experience = tracker.get_slot('activity_experience')
                activity_experience_mod = tracker.get_slot('activity_experience_mod')
                data_tuple = (sessions_done,
                              attention_check_list, attention_check_2_list, activity_index_list,
                              action_index_list, state, activity_experience, 
                              activity_experience_mod,
                              action_type_index_list, action_planning_answer, 
                              reflection_answer, user_id)
                sqlite_query = """UPDATE users SET sessions_done = ?, attention_check_list = ?, attention_check_2_list = ?, activity_index_list = ?, action_index_list = ?, state_4 = ?, activity_experience4 = ?, activity_experience_mod4 = ?, action_type_index_list = ?, action_planning_answer4 = ?, reflection_answer4 = ? WHERE id = ?"""
                link = "https://app.prolific.co/submissions/complete?cc=3B91AA04"
                
           
            else:
                # error happened
                session_saved = False
                
            # Something went wrong in the handling of the specific session
            if len(link) == 0:
                session_saved = False
            
            cursor.execute(sqlite_query, data_tuple)
            sqliteConnection.commit()
            cursor.close()

        except sqlite3.Error as error:
            session_saved = False
            print("Error while connecting to sqlite", error)
        finally:
            if (sqliteConnection):
                sqliteConnection.close()
                print("The SQLite connection is closed")
                
        # connection closed
        
        return [SlotSet("session_saved", session_saved),
                SlotSet("prolific_link", link)]

# To save data gathered until the effort response
# This means the mood and the effort answer are saved as well
# as the user satisfaction in sessions 3 and 5.
# Only for sessions 2-5
class ActionSaveSessionEffort(Action):
    def name(self):
        return "action_save_session_effort"

    async def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # get user ID
        metadata = extract_metadata_from_tracker(tracker)
        user_id = metadata['userid']
        
        # Load slot values
        mood = tracker.get_slot('mood')
        
        # create db connection
        try:
            sqliteConnection = sqlite3.connect(DATABASE_PATH)
            cursor = sqliteConnection.cursor()
            sqlite_select_query = """SELECT * from users WHERE id = ?"""
            cursor.execute(sqlite_select_query, (user_id,))
            data = cursor.fetchall()
            
            # we are in session 2
            if data[0][1] == 1:
                
                mood_list = '|'.join([data[0][2], mood])
                reward = tracker.get_slot('reward')
                data_tuple = (mood_list, reward, user_id)
                
                sqlite_query = """UPDATE users SET mood_list = ?, reward_list = ? WHERE id = ?"""
            
            # we are in session 3
            elif data[0][1] == 2:
                mood_list = data[0][2].split('|')
                mood_list.append(mood)
                mood_list = '|'.join(mood_list)
                reward_list = '|'.join([data[0][7], tracker.get_slot('reward')])
                satisf = tracker.get_slot('user_satisfaction')
                data_tuple = (mood_list, reward_list, satisf, user_id)
                
                sqlite_query = """UPDATE users SET mood_list = ?, reward_list = ?, user_satisfaction2 = ? WHERE id = ?"""
            
            # we are in session 4
            elif data[0][1] == 3:
                mood_list = data[0][2].split('|')
                mood_list.append(mood)
                mood_list = '|'.join(mood_list)
                reward_list = data[0][7].split('|')
                reward_list.append(tracker.get_slot('reward'))
                reward_list = '|'.join(reward_list)
                data_tuple = (mood_list, reward_list, user_id)
                sqlite_query = """UPDATE users SET mood_list = ?, reward_list = ? WHERE id = ?"""
                
            # we are in session 5, i.e. the last session
            elif data[0][1] == 4:
                satisf = tracker.get_slot('user_satisfaction')
                mood_list = data[0][2].split('|')
                mood_list.append(mood)
                mood_list = '|'.join(mood_list)
                reward_list = data[0][7].split('|')
                reward_list.append(tracker.get_slot('reward'))
                reward_list = '|'.join(reward_list)
                data_tuple = (mood_list, reward_list, satisf, user_id)
                sqlite_query = """UPDATE users SET mood_list = ?, reward_list = ?, user_satisfaction4 = ? WHERE id = ?"""
            
            cursor.execute(sqlite_query, data_tuple)
            sqliteConnection.commit()
            cursor.close()

        except sqlite3.Error as error:
            print("Error while connecting to sqlite when saving after effort response", error)
        finally:
            if (sqliteConnection):
                sqliteConnection.close()
                print("The SQLite connection is closed")
                
        # connection closed
        return []