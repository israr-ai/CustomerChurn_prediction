## Customer Churn Prediction Web App (Machine Learning + Flask)

This project is a Customer Churn Prediction System built using Machine Learning (Random Forest Classifier) and deployed using a Flask Web Application.

It predicts whether a telecom customer will Churn (Leave the service) or Stay, based on important customer features like:

Tenure

Monthly Charges

Total Charges

Contract Type

Payment Method

The app also supports CSV file upload to predict churn for multiple customers at once.

## Project Features

✅ Predict churn for a single customer (Form Input)
✅ Predict churn for multiple customers (CSV Upload Feature)
✅ RandomForest ML model training
✅ Label Encoding + Standard Scaling
✅ Shows churn probability (%)
✅ Flask-based clean web interface
✅ Model saved using joblib

## Machine Learning Model Details

Algorithm Used: RandomForestClassifier

Preprocessing:

LabelEncoder (Contract, PaymentMethod)

StandardScaler (Scaling numeric values)

Dataset Used: Telco Customer Churn Dataset (Kaggle)

## Dataset Source

Dataset is taken from Kaggle:

📌 Telco Customer Churn Dataset
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Loaded using kagglehub in training script.

## Technologies Used

Python 🐍

Pandas

NumPy

Scikit-learn

Flask

Joblib

HTML / CSS (Frontend)

KaggleHub (Dataset Loader)

## 📂 Project Structure
```
Customer-Churn-Prediction/
│── app.py
│── train_model.py
│── model.pkl
│── scaler.pkl
│── contract_encoder.pkl
│── payment_encoder.pkl
│── requirements.txt
│── templates/
│     │── index.html
│     │── result.html
│     │── upload.html
│── static/
│     └── style.css
│── README.md

```
## Installation & Setup
# 1️⃣ Clone Repository
```
git clone https://github.com/your-username/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

```
# 2️⃣ Create Virtual Environment
```
python -m venv env

```

# 3️⃣ Activate Virtual Environment
```
windows
env\Scripts\activate

Mac/Linux:
source env/bin/activate

```
# 4️⃣ Install Dependencies
```
pip install -r requirements.txt

```
## ▶️ Run Flask App
```
python app.py

```
## Input Features Used in Prediction
The model uses only these features:

Feature	Description
tenure	Number of months customer stayed
MonthlyCharges	Monthly bill amount
TotalCharges	Total bill amount
Contract	Contract type (Month-to-month, One year, Two year)
PaymentMethod	Payment method used

🧑‍💻 Author

👤 Israr Shekh
💻 Aspiring ML Engineer / Data Scientist
🌐 GitHub: https://github.com/israr-ai