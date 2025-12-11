from pymongo import ASCENDING, DESCENDING

def create_indexes():
    # Behavior logs: index on student_id for fast per-student queries
    behavior_logs.create_index([("student_id", ASCENDING), ("timestamp", DESCENDING)], name="idx_student_time")


    # Behavior logs: index on timestamp for time-based queries / range queries
    behavior_logs.create_index([("timestamp", DESCENDING)], name="idx_timestamp")

    # Scores: index on student_id 
    scores.create_index([("student_id", ASCENDING)], name="idx_scores_student_id")

    # Attendance: index on date 
    attendance.create_index([("date", ASCENDING)], name="idx_attendance_date")

    print("Indexes ensured.")
