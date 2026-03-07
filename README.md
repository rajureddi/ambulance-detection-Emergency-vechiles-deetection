

---

# 🚑 Emergency Vehicle Detection System

This project uses **YOLO (You Only Look Once)** models to detect:

1. **Ambulances**
2. **Emergency Vehicles** (Ambulance, Fire Truck, Police,Army)

The system supports **image and video detection** through a Gradio interface.

---

## 📌 Project Overview

This application performs real-time object detection using two trained YOLO models:

| Model                       | Purpose                                  |
| --------------------------- | ---------------------------------------- |
| **Ambulance Model**         | Detects ambulance-specific classes       |
| **Emergency Vehicle Model** | Detects multiple emergency vehicle types |

The interface allows users to upload images or videos and visualize detection results with bounding boxes and labels.

---

## 🧠 Models Used

### ✅ 1. Ambulance Detection Model

Detects:

* ambulance
* ambulance_108
* ambulance_SOL
ambulance_model.pt --> use this model only for ambulance
---

### ✅ 2. Emergency Vehicle Detection Model

Detects:

* ambulance
* fire_truck
* police
emergency_model.pt --> use this model for emergency model
(Depending on training dataset)

---

## ⚙️ Requirements

Install dependencies:

```bash
pip install ultralytics gradio opencv-python numpy
```

---

## 🚀 How to Run

### ▶️ Start the Application

```bash
python app.py
```

Gradio will launch a browser interface.

---

## 🖼️ Image Detection

1. Upload an image
2. Click **Detect**
3. View bounding boxes & labels

---

## 🎬 Video Detection

1. Upload a video
2. Click **Detect in Video**
3. Processed video will display

---

## 🎯 Detection Logic

The system filters detections using:

* **Confidence Threshold**
* **Class-Based Filtering**

Only relevant emergency vehicle classes are visualized.

Example emergency classes:

```python
EMERGENCY_CLASSES = [
    "ambulance",
    "ambulance_108",
    "ambulance_SOL",
    "fire_truck",
    "police"
]
```

---

## ⚠️ Important Notes

### ✅ Confidence Threshold

Detection stability depends heavily on:

```python
CONFIDENCE_THRESHOLD = 0.50
```

Higher values → fewer false positives
Lower values → more detections but noisier

---

### ✅ Dataset Design Impact

If your dataset contains component classes like:

* ambulance_lamp
* police_lamp
* firewriting

Expect possible detection confusion.

For production systems, vehicle-level classes are preferred.

---

## 🛠️ Troubleshooting

### ❌ No Detections?

✔ Lower confidence threshold:

```python
CONFIDENCE_THRESHOLD = 0.25
```

---

### ❌ Wrong Objects Detected?

✔ Check dataset balance
✔ Improve negative samples
✔ Reduce component-level labels

---

### ❌ Slow Video Processing?

✔ Reduce image size:

```python
model.predict(frame, imgsz=416)
```

---

## 📈 Future Improvements

Possible upgrades:

* Object Tracking (IDs)
* Emergency Vehicle Priority Logic
* Detection Confidence Smoothing
* Real-Time Camera Feed
* Traffic Signal Integration


**for model files contact me** 
## REFERENCES

Developed for **Emergency Vehicle Detection & Smart Traffic Applications**

* Kaggle ambulance detection  - https://www.kaggle.com/datasets/amrutasalagare/vehicle-dataset/data

* Roboflow emergency vechile dataset - https://universe.roboflow.com/avanthika-s-nfpex/indian-emergency-vehicles

