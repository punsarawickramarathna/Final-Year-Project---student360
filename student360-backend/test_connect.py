from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()
uri = os.getenv("MONGODB_URI")
client = MongoClient(uri)
db = client.get_default_database()  # if DB name is in URI, else db = client['student360']

print("Databases:", client.list_database_names())  # Should show 'admin' etc.
# Insert a document
doc = {"student_id":"IT2021-001","behavior":"note_taking","timestamp":"2025-03-10T09:05:03"}
res = db["behavior_logs"].insert_one(doc)
print("Inserted id:", res.inserted_id)
print("Sample doc:", db["behavior_logs"].find_one({"_id": res.inserted_id}))
