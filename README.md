# Smart Parking Monitoring and Availability Prediction System

### KyungDong University Global Campus

---

## 📖 Project Overview

This project focuses on designing and developing an intelligent parking system that can **detect and predict parking availability** in real time within KyungDong University Global Campus.

The system leverages **Computer Vision and Machine Learning techniques** to analyze parking spaces using camera input and provide accurate occupancy information.

---

## 🎯 Problem Statement

Existing platforms such as KakaoMap and Kakao T rely mainly on static or GPS-based data and do not provide accurate real-time parking availability, especially for small or private parking areas.

👉 Our goal is to:

- Detect parking occupancy in real time
- Predict future availability
- Provide a smart and scalable solution for campus environments

---

## 👥 Team Members

- ALINGILYA CRISPIN NJEWA (2517116)
- Rai Chhenwi Hang (2517104)
- Agot John Ngor Majok (2517047)
- Hellah Audrey (2417263)
- Le Tri Thanh (2517105)
- Tharu Roshan (2517120)

---

## 👨‍🏫 Supervisor

**Dr. Ghulam Sarwar**

---

## 🎓 Program

Bachelor of Science in Smart Computing (Spring 2026)  
Department of Smart Computing

---

## 🧠 System Approach

### 📥 Input

- CCTV Camera / Phone / Video feed (CCTV preferred)

### ⚙️ Processing

1. Detect parking spaces
2. Analyze each space
3. Classify:
   - 🟢 Free
   - 🔴 Occupied

### 📤 Output

- Real-time dashboard or mobile application
- Parking availability visualization

---

## 🔬 Technical Methodology

### Step 1: Parking Space Detection

- Define parking regions (bounding boxes)
- Check if a vehicle is present in each region

### Step 2: Vehicle Detection

- Use:
  - YOLOv5 / YOLOv8
  - CNN Classifier

### Step 3: Intelligent Analysis

- Car detection using YOLO
- Segmentation / clustering for space identification

---

## 📊 Data Collection

- Existing datasets:
  - PKLot
  - CNRPark
  - Kaggle datasets
- Custom dataset:
  - Images/videos from KDU Global Campus

---

## 🎯 Target Area

KDU Global Campus parking spaces

---

## 🚀 Key Features

- Real-time parking detection
- AI-based availability prediction
- Works on small/private parking areas
- Scalable for smart campus deployment

---

## 💡 Innovation

Unlike existing systems, our solution provides:

- Visual-based detection using cameras
- Real-time accuracy
- Predictive analytics for future parking availability
- Deployment flexibility for campuses and private areas

---

## 📂 Project Structure (Planned)
