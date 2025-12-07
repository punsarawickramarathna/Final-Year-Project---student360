
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "student360")

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]

behavior_logs = db["behavior_logs"]
scores = db["scores"]
attendance = db["attendance"]
users = db["users"]

def create_indexes():
    behavior_logs.create_index("student_id")
    behavior_logs.create_index("timestamp")
    scores.create_index("student_id")
    attendance.create_index("date")

