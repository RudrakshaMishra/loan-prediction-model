import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pickle

# ------------------------------
# Load dataset
# ------------------------------
df = pd.read_csv("loanfile.csv")

# ------------------------------
# Create target variable
# ------------------------------
df["target"] = (df["loan_status"] == "Charged Off").astype(int)

# Remove unused columns
drop_cols = [
    "loan_status",
    "funded_amnt",
    "funded_amnt_inv",
    "installment",
    "grade",
    "sub_grade",
]

df = df.drop(columns=drop_cols, errors="ignore")

# ------------------------------
# Keep only user-friendly columns
# ------------------------------
keep_cols = [
    "loan_amnt",
    "term",
    "int_rate",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "purpose",
]

df = df[keep_cols + ["target"]]

# ------------------------------
# Handle missing values
# ------------------------------
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna("Unknown")
    else:
        df[col] = df[col].fillna(df[col].median())

# ------------------------------
# Encode categorical variables
# ------------------------------
encoders = {}
for col in ["term", "emp_length", "home_ownership", "purpose"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ------------------------------
# Split dataset
# ------------------------------
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# FIX IMBALANCED DATASET USING SMOTE
# ------------------------------
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# ------------------------------
# Train a balanced Random Forest
# ------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

print("Model Accuracy:", model.score(X_test, y_test))

# ------------------------------
# Save model + encoders + columns
# ------------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(encoders, open("label_encoders.pkl", "wb"))
pickle.dump(list(X.columns), open("columns.pkl", "wb"))

print("Model files saved successfully.")
