# 🍷 Wine Classification Web Application

An end-to-end Machine Learning web application that classifies wine into different categories based on its chemical characteristics. The project uses a trained Scikit-learn classification model deployed with FastAPI and a modern HTML/CSS frontend.

---

## 📌 Project Overview

This project demonstrates the complete Machine Learning workflow:

- Data Preprocessing
- Model Training
- Model Evaluation
- Model Serialization
- FastAPI Backend Development
- Interactive Web Interface
- API Deployment Ready

The user enters 13 chemical properties of a wine sample, and the application predicts the wine class along with the prediction confidence.

---

## 🚀 Features

- Machine Learning Classification Model
- FastAPI REST API
- Responsive HTML & CSS User Interface
- Prediction Confidence Score
- Model Version Display
- Health Check Endpoint
- Input Validation using Pydantic
- Modular Project Structure

---

## 📊 Dataset

The project uses the Wine Dataset from Scikit-learn.

Dataset contains **178 samples** with **13 numerical features** representing the chemical composition of different wine cultivars.

Features include:

- Alcohol
- Malic Acid
- Ash
- Alcalinity of Ash
- Magnesium
- Total Phenols
- Flavanoids
- Nonflavanoid Phenols
- Proanthocyanins
- Color Intensity
- Hue
- OD280/OD315 of Diluted Wines
- Proline

Target Classes:

- Class 0
- Class 1
- Class 2

---

## 🛠 Technologies Used

- Python
- FastAPI
- Scikit-learn
- NumPy
- Joblib
- HTML5
- CSS3
- Jinja2
- Uvicorn

---

## 📁 Project Structure

```
Wine-Classification-FastAPI
│
├── templates
│   └── index.html
│
├── static
│   └── style.css
│
├── .gitignore
├── Demo Video Wine
├── README.md
├── main.py
├── model.pkl
├── requirements.txt
│
└── train.ipynb
```

---

## ⚙ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Wine-Classification-FastAPI.git
```

Go to project folder

```bash
cd Wine-Classification-FastAPI
```

Create virtual environment

```bash
python -m venv venv
```

Activate virtual environment

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run FastAPI

```bash
uvicorn main:app --reload
```

Open your browser

```
http://127.0.0.1:8000
```

---

## 🔗 API Endpoints

### Home

```
GET /
```

Loads the web interface.

### Prediction

```
POST /predict
```

Returns:

- Predicted Class
- Predicted Label
- Confidence Score
- Model Version

### Health Check

```
GET /health
```

Returns application health status.

---

## 🎥 Demo Video

A complete demonstration of the application is available in this repository.

The demo video showcases:

- User-friendly Web Interface
- Input of Wine Features
- Real-time Prediction
- Prediction Confidence Score
- FastAPI Backend Integration

📂 **Demo Video:** `demo.mp4`

---

## 🎯 Future Improvements

- Docker Deployment
- Cloud Deployment (Render / Railway)
- Model Monitoring
- User Authentication
- Prediction History
- Dark Mode Interface

---

## 👩‍💻 Author

**Palwasha Mushtaq**

MBA Finance | Data Science Enthusiast

Passionate about Machine Learning, AI, Data Analytics, FastAPI and Python Development.

---

## ⭐ If you like this project

Give this repository a ⭐ on GitHub.
