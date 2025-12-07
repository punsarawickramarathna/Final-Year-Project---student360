# main.py
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from db import behavior_logs, scores, attendance, users
from rating_engine import calculate_simple_score
from auth import create_access_token, verify_token
from security import hash_password, verify_password
from db import create_indexes
from fastapi import Query
from fastapi.responses import StreamingResponse
import csv
import io

app = FastAPI(title="Student360 Backend")
security = HTTPBearer()
create_indexes()

class BehaviorLogIn(BaseModel):
    student_id: str
    behavior: str
    timestamp: Optional[str] = None
    duration: Optional[float] = None
    confidence: Optional[float] = None
    track_id: Optional[str] = None

class UserIn(BaseModel):
    username: str
    password: str

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload


# --- Auth (POC)
@app.post("/auth/register")
def register(user: UserIn):
    existing = users.find_one({"username": user.username})
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")

    hashed_pw = hash_password(user.password)
    users.insert_one({
        "username": user.username,
        "password": hashed_pw
    })

    return {"message": "User registered securely"}

@app.post("/auth/login")
def login(user: UserIn):
    u = users.find_one({"username": user.username})
    if not u:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not verify_password(user.password, u["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}


# --- Core endpoints
@app.post("/behavior-logs", summary="Upload a batch of behavior logs")
def upload_behavior_logs(logs: List[BehaviorLogIn], current_user=Depends(get_current_user)):
    inserted = []
    for l in logs:
        ts = l.timestamp or datetime.utcnow().isoformat()
        doc = {
            "student_id": l.student_id,
            "behavior": l.behavior,
            "timestamp": ts,
            "duration": l.duration,
            "confidence": l.confidence,
            "track_id": l.track_id
        }
        res = behavior_logs.insert_one(doc)
        doc["_id"] = str(res.inserted_id)
        inserted.append(doc)
    return {"inserted": len(inserted), "documents": inserted}



@app.get("/student/{student_id}/logs", summary="Get logs for a student with paging")
def get_student_logs(
    student_id: str,
    current_user=Depends(get_current_user),
    limit: int = Query(20, ge=1, le=100),     # default 20, max 100
    skip: int = Query(0, ge=0),               # default 0
    sort_desc: bool = Query(True)
):
    sort_order = [("timestamp", -1)] if sort_desc else [("timestamp", 1)]
    total = behavior_logs.count_documents({"student_id": student_id})
    cursor = behavior_logs.find({"student_id": student_id}).sort(sort_order).skip(skip).limit(limit)
    docs = []
    for d in cursor:
        d["_id"] = str(d["_id"])
        docs.append(d)
    return {"student_id": student_id, "total": total, "limit": limit, "skip": skip, "logs": docs}


@app.get("/rating/{student_id}", summary="Compute and save rating")
def get_rating(student_id: str, current_user=Depends(get_current_user)):
    docs = list(behavior_logs.find({"student_id": student_id}))
    if not docs:
        raise HTTPException(status_code=404, detail="No logs for student")
    score, category = calculate_simple_score(docs)
    scores.update_one({"student_id": student_id}, {"$set": {
        "student_id": student_id, "score": score, "category": category, "updated": datetime.utcnow().isoformat()
    }}, upsert=True)
    return {"student_id": student_id, "score": score, "category": category}

@app.get("/attendance/{date}", summary="Attendance list for date")
def get_attendance_for_date(date: str, current_user=Depends(get_current_user)):
    start = f"{date}T00:00:00"
    end = f"{date}T23:59:59"
    pipeline = [
        {"$match": {"timestamp": {"$gte": start, "$lte": end}}},
        {"$group": {"_id": "$student_id"}}
    ]
    rows = list(behavior_logs.aggregate(pipeline))
    present = [r["_id"] for r in rows]
    return {"date": date, "present": present}

@app.get("/class/{class_id}/summary", summary="Class summary aggregated by behavior")
def get_class_summary(class_id: str, current_user=Depends(get_current_user)):
    # match student_id that starts with class_id (regex ^class_id)
    pipeline = [
        {"$match": {"student_id": {"$regex": f"^{class_id}"}}},
        {"$group": {"_id": "$behavior", "count": {"$sum": 1}}},
        {"$project": {"behavior": "$_id", "count": 1, "_id": 0}},
        {"$sort": {"count": -1}}
    ]
    results = list(behavior_logs.aggregate(pipeline))
    # Also compute total events and top offenders optionally
    total_events = sum(r["count"] for r in results) if results else 0

    # Optional: top students with most negative behaviors (example)
    negative_behaviors = ["sleeping", "using_phone", "turning_around", "yawning"]
    pipeline_top_bad = [
        {"$match": {"student_id": {"$regex": f"^{class_id}"}, "behavior": {"$in": negative_behaviors}}},
        {"$group": {"_id": "$student_id", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 10},
        {"$project": {"student_id": "$_id", "count": 1, "_id": 0}}
    ]
    top_offenders = list(behavior_logs.aggregate(pipeline_top_bad))

    return {"class_id": class_id, "total_events": total_events, "by_behavior": results, "top_offenders": top_offenders}

@app.on_event("startup")
def on_startup():
    create_indexes()


def csv_generator_from_cursor(cursor, fieldnames):
    # yields CSV lines
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    yield output.getvalue()
    output.seek(0)
    output.truncate(0)
    for doc in cursor:
        # sanitize / flatten
        row = {k: doc.get(k, "") for k in fieldnames}
        # convert ObjectId and nested fields to string
        if "_id" in doc:
            row["_id"] = str(doc["_id"])
        writer.writerow(row)
        yield output.getvalue()
        output.seek(0)
        output.truncate(0)

@app.get("/export/student/{student_id}/logs.csv", summary="Export student's logs as CSV")
def export_student_logs_csv(student_id: str, current_user=Depends(get_current_user)):
    cursor = behavior_logs.find({"student_id": student_id}).sort("timestamp", -1)
    fieldnames = ["_id", "student_id", "behavior", "timestamp", "duration", "confidence", "track_id"]
    generator = csv_generator_from_cursor(cursor, fieldnames)
    filename = f"logs_{student_id}.csv"
    return StreamingResponse(generator, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={filename}"})

# Example: export class logs (optionally use date filter parameters)
@app.get("/export/class/{class_id}/logs.csv", summary="Export class logs as CSV")
def export_class_logs_csv(class_id: str, current_user=Depends(get_current_user)):
    cursor = behavior_logs.find({"student_id": {"$regex": f"^{class_id}"}}).sort("timestamp", -1)
    fieldnames = ["_id", "student_id", "behavior", "timestamp", "duration", "confidence", "track_id"]
    generator = csv_generator_from_cursor(cursor, fieldnames)
    filename = f"logs_{class_id}.csv"
    return StreamingResponse(generator, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={filename}"})
