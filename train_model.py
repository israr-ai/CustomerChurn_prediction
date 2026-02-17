import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier


import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "blastchar/telco-customer-churn",
    file_path
)

print(df.head())
print(df.shape)



df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.fillna(df.mean(numeric_only=True), inplace=True)

# Only use these columns
df = df[["tenure", "MonthlyCharges", "TotalCharges", "Contract", "PaymentMethod", "Churn"]]

# Encode categorical
le_contract = LabelEncoder()
le_payment = LabelEncoder()

df["Contract"] = le_contract.fit_transform(df["Contract"])
df["PaymentMethod"] = le_payment.fit_transform(df["PaymentMethod"])

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

X = df.drop("Churn", axis=1)
y = df["Churn"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_contract, "contract_encoder.pkl")
joblib.dump(le_payment, "payment_encoder.pkl")

print("Model saved successfully!")

# Updated app.py (Matching Features)
from flask import Flask, render_template, request
import joblib
import pandas as pd

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

    return render_template("result.html", prediction=result, probability=round(prob*100, 2))

if __name__ == "__main__":
    app.run(debug=True)