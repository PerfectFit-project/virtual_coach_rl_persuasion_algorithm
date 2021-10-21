# ca_support_quit_smoking

This is the code for a conversational agent that suggests preparatory activities for smoking cessation and becoming more physically active in 5 separate sessions. This conversational agent was used in a longitudinal study with more than 500 participants with the goal of testing a personalized reinforcement learning-based persuasion algorithm.

## Experiment Flow

- Recruitment in Prolific
- Pre-screening
- Pre-qeusttionnaire
- 5 conversational sessions
- Post-questionnaire

## Dialog Flow

The figure below visualizes the structure of the 5 conversational sessions.

<img src = "Images/Dialog_Flow.jpg" width = "400" title="Dialog Flow">

## System Architecture

### Frontend
The frontend is a simple html-page that makes use of [Rasa Webchat](https://github.com/botfront/rasa-webchat) 0.11.12.

It is expected that the user ID is provided as a URL-parameter, e.g. http://<IP_address>:5005/?userid=JohnDoe if the frontend is running on port 5005. This user ID is extracted and sent to the backend to link multiple sessions of the same user.

Files:
- index.html: html-page if the conversational agent runs locally.
- Frontend: contains the html-pages for the 5 sessions if the conversational agent runs on a server. The frontends are run within Docker containers. This folder also contains the necessary Dockerfile. After building a Docker image for a frontend, a Docker container can be run via `docker run -d -p <port, e.g. 5005>:80 <imageName>`.
- connectors: contains the file "socketChannel.py." This file is needed to connect the frontend to the backend.

### Database
An sqlite database (sqlite3) is created within the custom action container. The database initialization script is in the "db_scripts"-folder and can be run via `python init_db.py` from within the running custom action container. This will create the database ":/tmp/chatbot.db" in the container.

There are several custom actions (in "actions.py") that read/write from/to this database.

Please note that the database is destroyed if one stops the custom action container, e.g. via `docker-compose down`. To save the database beyond the lifetime of the custom action container, copy it outside the container before stopping the container.

### Backend

The main component is a conversational agent trained in Rasa 2.0.2.

Files:
- actions: custom actions, e.g. to read from a database.
- models: contains trained models.
- config.yml: configuration for the training of the agent.
- data: contains files to specify e.g. the training stories on which the agent is trained.
- domain.yml: utterances, slots, etc.
- endpoints.yml: defines the endpoints of the conversational agent. 

## Other Components

### Preparatory Activities

The preparatory activities are provided in the files "Activities.csv"/"Activities.xlsx." These files contain formulations of the activities for different places in the sessions. E.g. there is a different activity formulation for the reminder message that people receive in Prolific and the session itself.

### Persuasion

There are multiple components to persuading people to do their suggested preparatory activities:
- In each session, a persuasive message is sent. Persuasive messages differ based on the persuasion type (Commitment, Consensus, Authority, Action Planning) as well as the preparatory activity they are used for. Moreover, there are multiple different messages for each combination of persuasion type and preparatory activity. All persuasive messages are in the file "all_messages.csv."
- In the case of the Commitment, Consensus and Authority persuasion types, a person is subsequently asked a reflective question. These reflective questions are given in the file "reflective_questions.csv."
- After each session, people receive a reminder message in Prolific. This message contains the formulation of the assigned activity, as well as a reminder question that depends on the persuasion type. All reminder questions can be found in the file "all_reminders.csv." The templates for the reminder messages for the first 4 and the last session are in the files "reminder_template.txt" and "reminder_template_last_session.csv," respectively. There are also templates for these messages in case no persuasion is given.

### Running the Agent on a Server with Rasa X and Docker

In our experiment, we hosted the conversational agent on a Google compute instance with Rasa X and Docker containers. As the required setup differs slightly from running the agent locally, the corresponding files are provided in the "server_stuff"-folder. This includes, for example, a Dockerfile for running the custom action code.

## License

Copyright (C) 2021 Delft University of Technology.

Licensed under the Apache License, version 2.0. See LICENSE for details.