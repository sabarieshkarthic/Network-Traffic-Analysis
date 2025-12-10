# 🛰️ Network Traffic Analyzer & Optimization System

A Streamlit-based real-time application for capturing, analyzing, classifying, and optimizing network traffic using machine learning models. The system leverages **Scapy**, **KMeans**, **Decision Trees**, and **XGBoost** to build a lightweight pipeline for monitoring bandwidth, jitter, and traffic types.

---

## 📌 Problem Statement
Modern networks require continuous monitoring to maintain Quality of Service (QoS) and prevent congestion. Low-level packet inspection is time‑consuming and noisy, making it difficult to identify actionable insights such as traffic type (VoIP, browsing, streaming) or resource bottlenecks.

This project solves that by:
- Automating packet capture
- Extracting statistical features
- Classifying traffic types
- Predicting bandwidth and jitter
- Suggesting optimization actions in real time

---

## 📄 Abstract
This tool captures packets in 10‑second windows and extracts nine interpretable features (burst rate, packet size, entropy, jitter, etc.). Using these features:
- **KMeans** clusters the traffic into 5 groups.
- Clusters are mapped to real-world classes (VoIP, Browsing, Streaming, Interactive TCP, Unknown).
- A **DecisionTreeClassifier** learns to classify future batches.
- Two **XGBoost models** predict **bandwidth** and **jitter**.

The Streamlit interface provides:
- Live metrics for every batch
- Automatic optimization actions
- Historical analytics (bandwidth trends, jitter trends, traffic distribution)

---

## 📊 Data Capture & Feature Extraction
Traffic data is collected from the local network using **Scapy**:
```
```
Each capture window is converted into a single feature vector:

| Feature | Description |
|--------|-------------|
| `burst_rate` | Packets per second |
| `avg_packet_size` | Mean size of packets |
| `protocol_ratio_TCP` | % of TCP packets |
| `protocol_ratio_UDP` | % of UDP packets |
| `size_scaled` | Min-max normalized packet size |
| `port_entropy` | Entropy of source ports |
| `inter_packet_gap` | Average time gap between packets |
| `jitter` | Std deviation of gaps |
| `byte_ratio_uplink_downlink` | Ratio of LAN uplink to downlink bytes |

These features are used for modeling traffic behavior.

---

## 🔍 Literature‑Inspired Approach

- Clustering to infer traffic types from unlabeled data
- Tree‑based models for interpretability
- Regression for continuous QoS indicators


---

## ⚙️ Methods & Implementation

### 1️⃣ Phase 1 — Unsupervised Learning
- Capture multiple batches
- Extract features
-Train KMeans and assign traffic types using simple heuristic rules
-(Streaming, VoIP, Browsing, Interactive TCP, Unknown).
- Train models:
  - Decision Tree → Classify traffic type
  - XGBoost → Predict bandwidth (Mbps)
  - XGBoost → Predict jitter (ms)
- Save models to disk using Joblib

### 2️⃣ Phase 2 — Monitoring
- Load trained models
- Capture live batches
- Predict traffic type, bandwidth, jitter
- Trigger suggested optimizations
- Update dashboard and logs

---

## ✅ Main Results
- Real-time traffic classification
- Bandwidth & jitter prediction with ML regressors
- Automated action suggestions
- Detailed analytics:
  - Bandwidth trend chart
  - Jitter trend chart
  - Traffic distribution (pie)
  - Bandwidth per traffic type (bar)

- Lightweight statistical features can classify traffic effectively.
- KMeans + rule-based labeling is a strong alternative when labeled data is absent.
- Decision Trees offer clarity for network operators.
- Jitter and Bandwidth prediction is useful for  stability monitoring.
- Real-time windowing gives actionable short-term QoS signals

---

## 🚀 App Features
- Live capture using Scapy
- Dynamic Streamlit UI
- Auto-updating charts
- Automatic ML model training and loading
- Two operational phases:
  - **Phase 1:** Learning
  - **Phase 2:** Monitoring

---







