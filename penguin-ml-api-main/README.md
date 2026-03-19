# 🐧 Penguin ML API — End-to-End ML Deployment

An end-to-end machine learning project that trains a classification model and deploys it as a production-ready API using **FastAPI + Docker + Cloud Deployment (Render)**.

---

## 🚀 Live Demo

- 🌐 **Web UI:** https://penguin-ml-api.onrender.com/ui  
- 🔎 **API Root:** https://penguin-ml-api.onrender.com  

> ⚠️ Hosted on Render free tier. First request may take ~30 seconds if idle.

---

## 📌 Project Overview

This project demonstrates the complete ML-to-production workflow:

- Data preprocessing & feature engineering  
- Model training (RandomForest – scikit-learn)  
- Model serialization (`model.pkl`)  
- REST API development with FastAPI  
- Docker containerization  
- Cloud deployment (Render)  
- Automatic redeployment via Git push  

It simulates how ML systems are built and served in real-world production environments.

---

## 🏗 Architecture

<!-- Replace the file name below with your actual architecture image -->


<img width="1100" height="476" alt="image" src="https://github.com/user-attachments/assets/709ff81c-7296-4e12-b0b8-9230451d8d15" />

---

## 🛠 Tech Stack

- **Language:** Python 3.11  
- **ML Model:** scikit-learn (RandomForest)  
- **API Framework:** FastAPI  
- **Server:** Uvicorn  
- **Containerization:** Docker  
- **Deployment:** Render (Cloud PaaS)  
- **Testing:** Pytest  

---

## 📂 Project Structure

```
penguin-ml-api/
│
├── artifacts/
│   └── model.pkl
│
├── src/
│   ├── app.py
│   └── models/
│       ├── train.py
│       ├── preprocess.py
│       └── predict.py
│
├── tests/
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🧠 API Endpoints

| Method   | Route      | Description |
|----------|-----------|------------|
| GET      | `/`       | API status |
| GET      | `/health` | Health check |
| POST     | `/predict`| Predict species (JSON) |
| GET/POST | `/ui`     | Web interface |

---

### 📥 Example JSON Request

```json
{
  "bill_length_mm": 43.2,
  "bill_depth_mm": 18.7,
  "flipper_length_mm": 195,
  "body_mass_g": 4200
}
Output : Adelie 
```
### 🎯 Target Prediction

The ML model predicts the **species of a penguin** based on physical measurements.

Supported prediction classes:

- **Adelie**
- **Chinstrap**
- **Gentoo**

The prediction is generated using the following input features:

- `bill_length_mm`
- `bill_depth_mm`
- `flipper_length_mm`
- `body_mass_g`
---

## 🖥 Run Locally

### 1️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 2️⃣ Start API server

```
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

Open in browser:

```
http://localhost:8000/ui
```

---

## 🐳 Run with Docker

### Build image

```
docker build -t penguin-ml-api .
```

### Run container

```
docker run -p 8000:8000 penguin-ml-api
```

---

## 🔄 Deployment

This project uses:

- GitHub (version control)
- Docker (container build)
- Render (automatic deployment from `main` branch)

Every push to `main` triggers a fresh deployment.

---

## 🧪 Testing

```
pytest
```

---

## 🖼 UI Preview

<!-- Replace the file name below with your actual UI screenshot -->

<img width="1737" height="1198" alt="image" src="https://github.com/user-attachments/assets/640704f1-018c-459c-99ef-6b4cb92b5432" />


---

## 📈 Why This Project Matters

This project demonstrates practical ML engineering skills:

✔ Model training pipeline  
✔ API development  
✔ Containerization  
✔ Cloud deployment  
✔ Public live service  

It reflects production-oriented backend + ML engineering capabilities beyond notebooks.

---

## 👨‍💻 Author

Built as part of a hands-on ML deployment portfolio project.

