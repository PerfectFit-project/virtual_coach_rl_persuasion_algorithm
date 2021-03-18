# Pull SDK image as base image
FROM rasa/rasa-sdk:2.0.0

# Use subdirectory as working directory
WORKDIR /app

# Copy actions requirements
COPY actions/requirements-actions.txt ./

# Change to root user to install dependencies
USER root

# Install extra requirements for actions code
RUN pip install -r requirements-actions.txt

# Copy actions code to working directory
COPY ./actions /app/actions
COPY all_messages.csv /app
COPY all_reminders.csv /app
COPY assignment.csv /app
COPY Activities.csv /app
COPY reminder_template_last_session.txt /app
COPY reminder_template.txt /app
COPY reflective_questions.csv /app
COPY x.txt /app
COPY email.txt /app
COPY Post_Sess_2 /app/Post_Sess_2

# Don't use root user to run code
USER 1001