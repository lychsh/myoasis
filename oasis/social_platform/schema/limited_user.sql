-- This is the schema definition for the limited user 
CREATE TABLE limited_user (
    user_id INTEGER PRIMARY KEY,
    limited_at DATETIME DEFAULT CURRENT_TIMESTAMP,  
    duration_time_steps INTEGER DEFAULT 72,      
    FOREIGN KEY(user_id) REFERENCES user(user_id)
);