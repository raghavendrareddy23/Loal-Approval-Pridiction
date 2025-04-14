# Loal-Approval-Pridiction

# 🧠 Loan Approval Prediction App

A machine learning project that predicts whether a loan application should be approved based on applicant details. The application is deployed using Streamlit and utilizes the **XGBoost** model, which showed the best performance among all trained models.

---

## 🚀 Live Demo

👉 [Try the App](https://raghavendrareddy23-loal-approval-pridiction-appstreamlit-o8jh4r.streamlit.app/)

---

## 📌 Features

- **Multi-model training** with:
  - ✅ XGBoost (Best performer)
  - RandomForest
  - Logistic Regression
  - Gradient Boosting
- **Model Evaluation Metrics**:
  - Accuracy
  - F1 Score
  - Confusion Matrix
  - Classification Report
- Real-time **loan approval prediction**
- Saved and loaded models using `joblib`
- Clean and interactive UI using **Streamlit**

---

## 🛠️ Technologies Used

- Python 🐍
- Streamlit 📊
- scikit-learn 🔍
- XGBoost 🌲
- imbalanced-learn (SMOTE) ⚖️
- Pandas, NumPy
- YAML (for config)
- Joblib (for model serialization)

---

## 📂 Project Structure

```bash
Loan-Approval-Prediction/
│
├── app/
│   └── streamlit.py           # Streamlit UI code
├── config/
│   └── config.yaml            # File path to dataset
├── data/
│   └── raw
|       └── loan_data.csv      # Loan data file 
├── model/
│   ├── model.pkl              # Best trained model
│   └── preprocessor.pkl       # Fitted preprocessing pipeline
│
├── src/
│   ├── data/
│   │   └── loadData.py        # Function to load data
│   ├── features/
│   │   └── preprocess.py      # Preprocessing pipeline
│   └── models/
|       └── predict.py         # Model Predicting Code
│       └── train.py           # Model training code
│
├── main.py                    # main file to preprocess and train model
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation

## 📊 Model Performance Summary

| Model                | Accuracy | F1 Score | Train Accuracy |
|---------------------|----------|----------|----------------|
| Logistic Regression | 86.36%   | 0.8714   | 88.50%         |
| Random Forest       | 91.04%   | 0.9126   | 97.08%         |
| Gradient Boosting   | 91.11%   | 0.9123   | 94.09%         |
| **XGBoost** (Best)  | **93.03%** | **0.9299** | **96.74%** |

## ✅ XGBoost was chosen due to its superior F1-score on validation data.

