import os, pickle, time, sys
import numpy as np
import pandas as pd

# ---------- helper ----------
def print_header(s):
    print("\n" + "="*10 + " " + s + " " + "="*10 + "\n")

# ---------- locate model file ----------
model_file = "model.pkl"
if not os.path.exists(model_file):
    print(f"ERROR: {model_file} not found in {os.getcwd()}")
    sys.exit(1)

stat = os.stat(model_file)
print_header("Model file info")
print("Path:", os.path.abspath(model_file))
print("Size (bytes):", stat.st_size)
print("Last modified:", time.ctime(stat.st_mtime))

# ---------- load model ----------
print_header("Loading model")
with open(model_file, "rb") as f:
    model = pickle.load(f)

print("Loaded object type:", type(model))
# print attributes if exist
for att in ("classes_", "n_estimators", "feature_importances_"):
    if hasattr(model, att):
        val = getattr(model, att)
        try:
            print(f"{att}:", np.array(val) if hasattr(val, "__len__") and len(val)<=20 else str(type(val)))
        except Exception as e:
            print(f"{att}: (error printing) {e}")

# ---------- load columns if present ----------
cols = None
if os.path.exists("columns.pkl"):
    with open("columns.pkl", "rb") as f:
        cols = pickle.load(f)
    print_header("Columns info (columns.pkl)")
    print("Columns count:", len(cols))
    print(cols[:50])

# ---------- load label encoders if present ----------
if os.path.exists("label_encoders.pkl"):
    with open("label_encoders.pkl", "rb") as f:
        le = pickle.load(f)
    print_header("Label encoders info")
    print("Encoders keys:", list(le.keys()))

# ---------- define extreme test cases ----------
def mk_row(loan_amnt, term, int_rate, emp_length, home_ownership, annual_inc, purpose, cols):
    # Create a dict of strings for categories, numbers for numeric
    d = {
        "loan_amnt": loan_amnt,
        "term": term,
        "int_rate": int_rate,
        "emp_length": emp_length,
        "home_ownership": home_ownership,
        "annual_inc": annual_inc,
        "purpose": purpose
    }
    df = pd.DataFrame([d])
    # if encoders exist, try to transform
    if os.path.exists("label_encoders.pkl"):
        with open("label_encoders.pkl","rb") as f:
            enc = pickle.load(f)
        for c in df.columns:
            if c in enc:
                try:
                    df[c] = enc[c].transform([df[c].iloc[0]])
                except Exception:
                    # unknown category -> fallback to 0
                    df[c] = 0
    # align to columns
    if cols is not None:
        df = df.reindex(columns=cols, fill_value=0)
    return df

test_cases = [
    ("Extreme1", mk_row(999999999999, "60 months", 30.0, "1 year", "RENT", 2000, "small_business", cols)),
    ("Extreme2", mk_row(500000, "60 months", 28.0, "1 year", "RENT", 8000, "small_business", cols)),
    ("Extreme3", mk_row(30000, "60 months", 29.5, "1 year", "RENT", 12000, "medical", cols)),
    ("Safe1", mk_row(15000, "36 months", 10.0, "10+ years", "OWN", 120000, "credit_card", cols)),
]

print_header("Test predictions")
for name, df in test_cases:
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)
            pred = model.predict(df)
            print(f"\n{name} -> pred: {pred[0]}, proba: {probs[0]}")
        else:
            pred = model.predict(df)
            print(f"\n{name} -> pred: {pred[0]}")
    except Exception as e:
        print(f"\n{name} -> ERROR running prediction: {e}")

# ---------- check on a few training rows if loanfile exists ----------
if os.path.exists("loanfile.csv"):
    print_header("Quick checks on original dataset samples")
    dff = pd.read_csv("loanfile.csv", nrows=2000)
    # try to find some charged off rows
    if "loan_status" in dff.columns:
        co = dff[dff["loan_status"]=="Charged Off"].head(5)
        print("Found Charged Off examples (first 5):", len(co))
        if len(co)>0:
            sample = co.iloc[0].to_dict()
            # Build input dict with same kept fields if columns available
            keys = ["loan_amnt","term","int_rate","emp_length","home_ownership","annual_inc","purpose"]
            sample_row = {k: sample.get(k, "") for k in keys}
            print("Example charged-off sample (raw):", sample_row)
            df_s = mk_row(sample_row["loan_amnt"], sample_row["term"], sample_row["int_rate"],
                         sample_row["emp_length"], sample_row["home_ownership"], sample_row["annual_inc"],
                         sample_row["purpose"], cols)
            try:
                print("Model prediction on that charged-off sample:", model.predict(df_s)[0],
                      "proba:", model.predict_proba(df_s)[0] if hasattr(model,"predict_proba") else "N/A")
            except Exception as e:
                print("Error predicting on sample:", e)
    else:
        print("loan_status not found in the sample csv")
