from flask import Flask, render_template, request
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
import os

app = Flask(__name__)

model_path = hf_hub_download(repo_id="haidar99r49r/LoanML", filename="model.pkl")
model = joblib.load(model_path)

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/model")
def model_form():
    return render_template("model_form.html")

@app.route('/predict', methods=['POST'])
def predict():
    person_age = float(request.form.get('field1'))
    person_education = int(request.form.get('field2'))
    person_income = float(request.form.get('field3'))
    person_emp_exp = int(request.form.get('field4'))
    loan_amnt = float(request.form.get('field5'))
    loan_int_rate = float(request.form.get('field6'))
    loan_percent_income = loan_amnt/person_income
    cb_person_cred_hist_length = float(request.form.get('field8'))
    credit_score = int(request.form.get('field9'))

    pldof = request.form.get('field10', '').strip().upper()
    previous_loan_defaults_on_file = int(pldof == 'YES')

    selected_loan_intent = request.form.get('loan_intent', '').lower()
    loan_intent_DEBTCONSOLIDATION = int(selected_loan_intent == 'debt_consolidation')
    loan_intent_EDUCATION = int(selected_loan_intent == 'education')
    loan_intent_HOMEIMPROVEMENT = int(selected_loan_intent == 'home_improvement')
    loan_intent_MEDICAL = int(selected_loan_intent == 'medical')
    loan_intent_PERSONAL = int(selected_loan_intent == 'personal')
    loan_intent_VENTURE = int(selected_loan_intent == 'venture')

    selected_home = request.form.get('home_ownership', '').lower()
    person_home_ownership_MORTGAGE = int(selected_home == 'mortgage')
    person_home_ownership_OWN = int(selected_home == 'own')
    person_home_ownership_RENT = int(selected_home == 'rent')

    feature_names = [
    'person_age', 'person_education', 'person_income', 'person_emp_exp',
    'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
    'credit_score', 'previous_loan_defaults_on_file',
    'loan_intent_DEBTCONSOLIDATION', 'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT',
    'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE',
    'person_home_ownership_MORTGAGE', 'person_home_ownership_OWN', 'person_home_ownership_RENT'
    ]

    features_df = pd.DataFrame([[
    person_age, person_education, person_income, person_emp_exp,
    loan_amnt, loan_int_rate, loan_percent_income, cb_person_cred_hist_length,
    credit_score, previous_loan_defaults_on_file,
    loan_intent_DEBTCONSOLIDATION, loan_intent_EDUCATION, loan_intent_HOMEIMPROVEMENT,
    loan_intent_MEDICAL, loan_intent_PERSONAL, loan_intent_VENTURE,
    person_home_ownership_MORTGAGE, person_home_ownership_OWN, person_home_ownership_RENT
    ]], columns=feature_names)

    y_prob = model.predict_proba(features_df)[0][1]
    threshold = 0.45
    y_pred_adj = int(y_prob >= threshold)

    prob_percent = f"{y_prob * 100:.2f}%"
    rejection_prob = f"{(1 - y_prob) * 100:.2f}%"

    if y_pred_adj == 1 and y_prob <= 0.7:
        result_message = f"{prob_percent} Accepted with further inspection from the team due to high risk."
    elif y_pred_adj == 1 and y_prob > 0.7:
        result_message = f"{prob_percent} Accepted."
    else:
        result_message = f"{rejection_prob} Rejected."

    return render_template('result.html', prediction=result_message)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 1000))
    app.run(host="0.0.0.0", port=port)