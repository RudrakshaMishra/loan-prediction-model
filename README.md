# 📘 Loan Default Prediction – Machine Learning + Flask Web App

A complete **Loan Default Prediction system** built using **Machine Learning**, **Flask**, and **Python**, designed to predict whether a borrower is likely to *repay* or *default* on a loan.

This project includes:

- A trained ML model (Random Forest)
- Cleaned and preprocessed data pipeline
- Oversampling and balancing techniques
- Rule-based risk overrides (for unrealistic inputs)
- Flask-based UI to input borrower details
- Probability-based decisions
- Custom approval thresholds (bank-style)
- Ready for deployment on Render/Railway

---

## 🚀 Features

### ✔ **Loan Prediction Web App**
A simple Flask interface where users input:

- Loan amount  
- Term  
- Interest rate  
- Employment length  
- Home ownership  
- Annual income  
- Loan purpose  

### ✔ **Outputs**
The model predicts:

- **Likely to be Paid**  
- **Likely to Default**  
- Probability of default  
- Threshold decision logic  
- Rule-based override messages  

### ✔ **Rule-Based Underwriting Layer**
Even if ML is unsure, the following are auto-rejected:

- Loan amount > 20× annual income  
- Extremely high interest rates with very low income  
- Large loans with low income  

### ✔ **Balanced Dataset Training**
The dataset includes:

- 100,000 random loans  
- +40,000 oversampled default loans  

This improves default prediction significantly.

---

## 📂 Project Structure

