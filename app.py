from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
contract_encoder = joblib.load("contract_encoder.pkl")
payment_encoder = joblib.load("payment_encoder.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    tenure = float(request.form["tenure"])
    monthly = float(request.form["MonthlyCharges"])
    total = float(request.form["TotalCharges"])
    contract = request.form["Contract"]
    payment = request.form["PaymentMethod"]

    contract_encoded = contract_encoder.transform([contract])[0]
    payment_encoded = payment_encoder.transform([payment])[0]

    input_data = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Contract": contract_encoded,
        "PaymentMethod": payment_encoded
    }])

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)[0]
    prob = model.predict_proba(scaled_data)[0][1]

    result = "Customer Will Churn ❌" if prediction == 1 else "Customer Will Stay ✅"

    return render_template("result.html",
                           prediction=result,
                           probability=round(prob*100, 2))

@app.route("/upload")
def upload_page():
    return render_template("upload.html")

@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    file = request.files["file"]
    df = pd.read_csv(file)

    df.drop("customerID", axis=1, inplace=True)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Keep only columns used in training
    df = df[["tenure", "MonthlyCharges", "TotalCharges", "Contract", "PaymentMethod"]]

    df["Contract"] = contract_encoder.transform(df["Contract"])
    df["PaymentMethod"] = payment_encoder.transform(df["PaymentMethod"])

    X_scaled = scaler.transform(df)

    preds = model.predict(X_scaled)

    churn_count = np.sum(preds == 1)
    stay_count = np.sum(preds == 0)

    return render_template("upload.html",
                           churn_count=int(churn_count),
                           stay_count=int(stay_count),
                           total=len(preds))

if __name__ == "__main__":
    app.run(debug=True)
