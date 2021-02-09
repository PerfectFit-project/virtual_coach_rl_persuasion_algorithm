import sqlite3

sqliteConnection = None
try:
    sqliteConnection = sqlite3.connect('chatbot.db')

    sqlite_create_table_query = '''CREATE TABLE IF NOT EXISTS users (
                                id TEXT PRIMARY KEY,
                                sessions_done INTEGER,
                                mood_list TEXT,
                                action_planning_answer0 TEXT,
                                action_planning_answer1 TEXT,
                                action_planning_answer2 TEXT,
                                action_planning_answer3 TEXT,
                                reward_list TEXT,
                                activity_experience1 TEXT,
                                activity_experience2 TEXT,
                                activity_experience3 TEXT,
                                activity_experience4 TEXT,
                                activity_experience_mod1 TEXT,
                                activity_experience_mod2 TEXT,
                                activity_experience_mod3 TEXT,
                                activity_experience_mod4 TEXT,
                                attention_check_list TEXT,
                                attention_check_2_list TEXT,
                                activity_index_list TEXT,
                                action_index_list TEXT,
                                state_0 TEXT,
                                state_1 TEXT,
                                state_2 TEXT,
                                state_3 TEXT,
                                state_4 TEXT,
                                action_type_index_list TEXT,
                                study_group INTEGER,
                                user_satisfaction2 TEXT,
                                user_satisfaction4 TEXT,
                                reflection_answer0 TEXT,
                                reflection_answer1 TEXT,
                                reflection_answer2 TEXT,
                                reflection_answer3 TEXT
                                )'''
								
    cursor = sqliteConnection.cursor()
    print("Successfully connected to SQLite")
    #cursor.execute('''DROP table users;''')
    cursor.execute(sqlite_create_table_query)
    sqliteConnection.commit()
    print("SQLite table created")
    data_tuple = ('111', 2, 'droopy|sad', "sdd", " ", " ", "4|10", "1|2", "2|3|1|4|1|3|1|0|3|0", "2|3", "2|0|1|2|1|3|3|0|3|0", "1")
    sqlite_query = """INSERT INTO users (id, sessions_done, mood_list, action_planning_answer0, attention_check_list, attention_check_2_list, activity_index_list, action_index_list, state_0, action_type_index_list, state_1, reward_list) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
    cursor.execute(sqlite_query, data_tuple)
    sqliteConnection.commit()
    data_tuple = ('222', 2, 'droopy|happy', "sdd", " ", " ", "2|9", "1|2", "0|4|1|4|1|2|1|1|3|0", "2|3","2|0|2|0|0|0|3|0|4|0", "0")
    sqlite_query = """INSERT INTO users (id, sessions_done, mood_list, action_planning_answer0, attention_check_list, attention_check_2_list, activity_index_list, action_index_list, state_0, action_type_index_list, state_1, reward_list) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""" 
    cursor.execute(sqlite_query, data_tuple) 
    sqliteConnection.commit()    
    #data_tuple = ('333', 2, 'droopy|happy', "sdd", " ", " ", "2|9", "1|2", "1|3|1|2|1|0|0|1|0|0", "1|1","4|1|1|0|1|0|2|0|3|0", "1")
    #sqlite_query = """INSERT INTO users (id, sessions_done, mood_list, action_planning_answer0, attention_check_list, attention_check_2_list, activity_index_list, action_index_list, state_0, action_type_index_list, state_1, reward_list) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""" 
    #cursor.execute(sqlite_query, data_tuple) 
    sqliteConnection.commit()    	
    cursor.close()

except sqlite3.Error as error:
    print("Error while creating a sqlite table", error)
finally:
    if (sqliteConnection):
        sqliteConnection.close()
        print("sqlite connection is closed")