# ca_support_quit_smoking

This is the code for a conversational agent that suggests preparatory activities for smoking cessation and becoming more physically active in 5 separate sessions.

## Experiment Flow

- Recruitment in Prolific
- Pre-screening
- Pre-qeusttionnaire
- 5 conversational sessions
- Post-questionnaire

## Dialog Flow

The figure below visualizes the structure of the 5 conversational sessions.

<img src = "Images/Dialog_Flow.PNG" width = "400" title="Dialog Flow">

## System Architecture

### Frontend
The frontend is a simple html-page that makes use of [Rasa Webchat](https://github.com/botfront/rasa-webchat) 0.11.12.

Files:
- index.html: html-page if the conversational agent runs locally.
- Frontend: contains the html-pages for the 5 sessions if the conversational agent runs on a server. The frontends are run within Docker containers. This folder also contains the necessary Dockerfile.
- connectors: contains the file socketChannel.py. This file is needed to connect the frontend to the backend.

### Database


### Backend


### Activities
The preparatory activities are provided in the files "Activities.csv"/"Activities.xlsx." These files contain formulations of the activities for different places in the sessions. E.g. there is a different activity formulation for the reminder message that people receive in Prolific and the session itself.

### Persuasion

There are multiple components to persuading people to do their suggested preparatory activities:
- In each session, a persuasive message is sent. Persuasive messages differ based on the persuasion type (Commitment, Consensus, Authority, Action Planning) as well as the preparatory activity they are used for. Moreover, there are multiple different messages for each combination of persuasion type and preparatory activity. All persuasive messages are in the file "all_messages.csv."
- In the case of the Commitment, Consensus and Authority persuasion types, a person is subsequently asked a reflective question. These reflective questions are given in the file "reflective_questions.csv."
- After each session, people receive a reminder message in Prolific. This message contains the formulation of the assigned activity, as well as a reminder question that depends on the persuasion type. All reminder questions can be found in the file "all_reminders.csv." The templates for the reminder messages for the first 4 and the last session are in the files "reminder_template.txt" and "reminder_template_last_session.csv," respectively.