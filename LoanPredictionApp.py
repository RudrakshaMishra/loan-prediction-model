from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

# CONFIG: threshold for default probability above which we REJECT
# Set to 0.15 (15%) by default — adjust to 0.10 or 0.05 for stricter policies
DEFAULT_THRESHOLD = 0.15

# Rule-based overrides (examples)
# If any rule returns True -> immediate default (reject)
def rule_override(row):
    try:
        loan_amnt = float(row.get("loan_amnt", 0))
        annual_inc = float(row.get("annual_inc", 0))
        int_rate = float(row.get("int_rate", 0))
    except Exception:
        return False

    # Rule 1: absurd loan-to-income ratio
    if annual_inc > 0 and loan_amnt / annual_inc > 20:  # more than 20x annual income
        return True

    # Rule 2: extremely high interest with very low income
    if int_rate >= 25 and annual_inc < 20000:
        return True

    # Rule 3: enormous loan with low income
    if loan_amnt > 200_000 and annual_inc < 30000:
        return True

    return False

# -------------------------
# Load model + encoders + columns
# -------------------------
print("Working directory:", os.getcwd())
print("Model path:", os.path.abspath("model/model.pkl") if os.path.exists("model/model.pkl") else os.path.abspath("model.pkl"))

# Adjust paths if you placed model files in folder "model"
model_path = "model/model.pkl" if os.path.exists("model/model.pkl") else "model.pkl"
enc_path = "model/label_encoders.pkl" if os.path.exists("model/label_encoders.pkl") else "label_encoders.pkl"
cols_path = "model/columns.pkl" if os.path.exists("model/columns.pkl") else "columns.pkl"

model = pickle.load(open(model_path, "rb"))
encoders = pickle.load(open(enc_path, "rb"))
columns = pickle.load(open(cols_path, "rb"))

app = Flask(__name__)

# -------------------------
# helper: clean and normalize form inputs
# -------------------------
def clean_input_dict(data):
    cleaned = {}
    for k, v in data.items():
        if v is None:
            cleaned[k] = v
            continue
        # trim whitespace for strings
        if isinstance(v, str):
            s = v.strip()
            # normalize some fields to expected canonical forms
            if k == "term":
                # ensure it's like "36 months" or "60 months"
                s = s.replace(" ", "")  # remove spaces like " 36 months" -> "36months"
                if "36" in s:
                    s = "36 months"
                elif "60" in s:
                    s = "60 months"
            if k == "emp_length":
                # normalize common tokens
                if "10" in s:
                    s = "10+ years"
                elif "< 1" in s or "<1" in s:
                    s = "1 year"
            cleaned[k] = s
        else:
            cleaned[k] = v
    return cleaned

# -------------------------
# Home Page
# -------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -------------------------
# Prediction Route
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    # Get input data from form (Flask's request.form returns ImmutableMultiDict)
    raw = request.form.to_dict()
    data = clean_input_dict(raw)

    # Quick rule override check (uses raw numeric strings)
    if rule_override(data):
        # keep values for form
        result = f"❌ Loan Rejected by rule-based override (loan/income or interest rules)."
        return render_template("index.html", prediction=result, **raw)

    # Convert to DataFrame
    df = pd.DataFrame([data])

    # Ensure numeric columns are numeric (coerce bad input to 0)
    for num_col in ["loan_amnt", "int_rate", "annual_inc"]:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce").fillna(0)

    # Apply label encoders safely (strip/canonical values already applied)
    for col in df.columns:
        if col in encoders:
            try:
                # encoders expect 1D array-like
                val = df.at[0, col]
                # if numeric accidentally, convert to string before transform
                if not isinstance(val, (int, float)):
                    enc_val = encoders[col].transform([str(val)])[0]
                else:
                    enc_val = encoders[col].transform([str(val)])[0]
                df[col] = enc_val
            except Exception:
                # unknown category -> fallback to most common category index 0
                df[col] = 0

    # Make sure all expected columns exist and in same order
    df = df.reindex(columns=columns, fill_value=0)

    # Predict class and probability
    try:
        raw_pred = model.predict(df)[0]
        pred = int(raw_pred)
    except Exception as e:
        return render_template("index.html", prediction=f"ERROR predicting: {e}", **raw)

    proba_default = None
    if hasattr(model, "predict_proba"):
        proba_default = float(model.predict_proba(df)[0][1])

    # Apply threshold logic (bank-style)
    if proba_default is not None and proba_default > DEFAULT_THRESHOLD:
        result = f"❌ Loan Likely to Default (probability: {proba_default:.2f}) - Rejected (threshold {DEFAULT_THRESHOLD:.2f})"
    else:
        result = f"✔ Loan Likely to Be Paid (probability: {proba_default:.2f}) - Approved (threshold {DEFAULT_THRESHOLD:.2f})"

    # Render with original raw values so inputs stay visible
    return render_template("index.html", prediction=result, **raw)

# -------------------------
# Run App
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
