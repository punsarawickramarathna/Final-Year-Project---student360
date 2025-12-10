---
# 🎓 **Student360 – University Student Behavior detection & Attendance Monitoring System**

Student360 is a full-stack, AI-powered system designed to automatically detect and rate student behaviors in classrooms/exam halls and generate attendance using face-recognition-based identification.

This system uses

* YOLOv8 Behavior Detection (sleeping, yawning, phone use, turning around, raising hand, note-taking)

* Custom CNN Face Recognition Model (trained using PyTorch from student photo dataset)

* FastAPI Backend (secure JWT auth + behavior logging + rating engine + attendance generation)

* MongoDB Database (Atlas cloud)

* Dashboard Frontend (React / Web App)
  
---
# 📌 **Project Features**

### ✅ **Behavior Detection**

Detects and logs:

* Sleeping on desk
* Yawning
* Using mobile phone
* Turning around (exam cheating)
* Raising hand
* Actively taking notes

Behavior logs are stored via FastAPI.

---

### ✅ **Face Recognition Attendance (My Contribution)**

A custom **CNN model trained with PyTorch**:

* Dataset structure:

  ```
  dataset/faces_raw/<student_id>/*.jpg
  ```
* Automatic face extraction & alignment (MTCNN)
* Train/Validation auto split
* Trained ResNet18 classifier
* Inference script integrated into backend for real-time attendance generation

---

### ✅ **Backend Services (My Contribution)**

Built using **FastAPI** + **MongoDB**, including:

* Secure JWT authentication
* Behavior log submission API
* Automatic student rating engine
* Attendance generation (from CNN model + timestamps)
* CSV export endpoints
* Indexed MongoDB collections (optimized)

---

# 🏗️ **System Architecture**

```
 YOLO Model → Behavior Logs API → MongoDB
              ↑
Student Video → Face Extractor → CNN Attendance Model → Attendance API
                                              ↑
                                        FastAPI Backend
```

---

# 📁 **Project Folder Structure**

```
student360/
├── models/                 # Saved CNN .pth model (ignored by git)
├── dataset/                # Student images (ignored by git)
│   ├── faces_raw/          # Raw images (per student)
│   ├── train/
│   ├── val/
│
├── scripts/                # MODEL TRAINING SCRIPTS (My part)
│   ├── extract_faces.py
│   ├── align_faces.py
│   ├── split_dataset.py
│   ├── train.py
│   ├── infer.py
│
├── real_test/              # Real classroom images (ignored)
├── real_aligned/           # Preprocessed faces (ignored)
├── student360-backend/
    │── main.py                 # FastAPI backend
    │── auth.py                 # JWT authentication
    │── security.py             # Password hashing
    │── rating_engine.py        # Behavior scoring logic
    │── db.py                   # MongoDB connection
    │── indexes.py              # MongoDB index creation
    │── test_connect.py         # MongoDB test script
    │── requirements.txt        
│── README.md
```

---

# ⚙️ **Installation Guide**

### **1️⃣ Clone the Repository**

```sh
git clone https://github.com/YOUR_USERNAME/student360.git
cd student360
```

---

### **2️⃣ Create & Activate Virtual Environment**

```sh
python -m venv venv
venv\Scripts\activate
```

---

### **3️⃣ Install Dependencies**

```sh
pip install -r requirements.txt
```

---

### **4️⃣ Configure Environment Variables**

Create a `.env` file:

```
MONGODB_URI=your_atlas_uri
DB_NAME=student360
JWT_SECRET=your_strong_secret_key
```

---

### **5️⃣ Run the Backend**

```sh
uvicorn main:app --reload
```

Open Swagger UI:

👉 **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

---

# 🧠 **CNN Attendance Model (My Part)**

### **A. Face Extraction**

```sh
python scripts/extract_faces.py --input student.jpg --out dataset/faces_raw/STUDENT_ID
```

### **B. Face Alignment**

```sh
python scripts/align_faces.py --src dataset/faces_raw --dst dataset/aligned
```

### **C. Auto Split Train/Val**

```sh
python scripts/split_dataset.py --src dataset/aligned --dst dataset
```

### **D. Train CNN Model**

```sh
python scripts/train.py --data_dir dataset --work_dir models --epochs 50 --batch 16
```

### **E. Test (Inference)**

```sh
python scripts/infer.py --img test.jpg --threshold 0.75
```

---

# 🔐 **Backend API Endpoints**

### **Auth**

| Method | Endpoint         | Description         |
| ------ | ---------------- | ------------------- |
| POST   | `/auth/register` | Register a new user |
| POST   | `/auth/login`    | Get JWT token       |

---

### **Behavior Logs**

| Method | Endpoint             | Description                   |
| ------ | -------------------- | ----------------------------- |
| POST   | `/behavior-logs`     | Upload batch of behavior logs |
| GET    | `/student/{id}/logs` | Get logs with pagination      |

---

### **Rating Engine**

| Method | Endpoint               | Description           |
| ------ | ---------------------- | --------------------- |
| GET    | `/rating/{student_id}` | Compute + save rating |

---

### **Attendance**

| Method | Endpoint                    | Description                   |
| ------ | --------------------------- | ----------------------------- |
| GET    | `/attendance/{date}`        | Auto attendance based on logs |
| GET    | `/class/{class_id}/summary` | Summary of behavior per class |

---

### **Export**

| Method | Endpoint                            | Description    |
| ------ | ----------------------------------- | -------------- |
| GET    | `/export/student/{id}/logs.csv`     | Logs as CSV    |
| GET    | `/export/class/{class_id}/logs.csv` | Class logs CSV |

---

# 🧮 **Behavior Rating Model**

```
sleeping       = -10
yawning        =  -5
using_phone    = -20
turning_around = -15
raise_hand     = +10
note_taking    = +5
```

Produces:

* **good**
* **average**
* **poor**

---

# 🚀 **Deployment Guide**

### Deploy backend to Render/Heroku/VPS:

```sh
uvicorn main:app --host 0.0.0.0 --port 8000
```

MongoDB Atlas works automatically.

---

# 👤 **My Contribution Summary**

### ✔ Built entire **FastAPI backend**

* Auth (JWT)
* Behavior Logs API
* Rating Engine
* Attendance Engine
* MongoDB indexing
* CSV exports
* Data models & database layer

### ✔ Built full **face-recognition attendance pipeline**

* Face extraction
* Alignment (MTCNN/facenet)
* Dataset generation
* CNN training (ResNet18)
* Inference engine
* Backend integration

### ✔ Designed **database schema & optimization**

---
