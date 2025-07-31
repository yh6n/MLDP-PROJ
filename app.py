import streamlit as st
import joblib
import pandas as pd

# Load model data
@st.cache_data
def load_model():
    # Just loading the model and its associated threshold and features
    # In production, make sure the model file path is correct and that it's properly versioned
    data = joblib.load("diabetes_good.pkl")  # Loading the model, threshold, and features
    return data["model"], data["threshold"], data["features"]

model, threshold, features = load_model()  # Assigning the model, threshold, and features

# Configurations
MODEL_FEATURES = [
    'HighBP', 'HighChol', 'BMI', 'Stroke', 'HeartDiseaseorAttack',
    'PhysActivity', 'HvyAlcoholConsump', 'GenHlth', 'DiffWalk', 'Sex', 'Age'
]

# Risk categories based on the predicted probability
RISK_LEVELS = {
    (0.0, 0.3): ("Low Risk", "low"),  # Low probability means low risk
    (0.3, 0.5): ("Medium Risk", "medium"),  # Mid-range risk
    (0.5, 1.0): ("High Risk", "high")  # Higher probabilities indicate high risk
}

# Adding custom styling for the app (custom background, form, etc.)
st.markdown("""
<style>
.stApp {
    background-image: url('https://artedi.com.mx/wp-content/uploads/2024/03/fachada-hospital-contemporaneo.jpg');
    background-size: cover;
    background-repeat: no-repeat;
    background-position: center center;
    min-height: 100vh;
    font-family: Arial, sans-serif;
    position: relative;
}
.stApp:before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0, 0, 0, 0.45);
    z-index: 0;
    pointer-events: none;
}
.stApp > * { position: relative; z-index: 1; }

.stForm {
    background: rgba(12,12,12,0.85);
    border-radius: 20px !important;
    box-shadow: 0 6px 15px rgba(0,0,0,0.16);
    padding: 40px 32px !important;
    margin: 42px auto 32px auto;
    max-width: 600px;
}
.stForm label, .stForm span, .stForm input, .stForm select, .stForm textarea,
.stForm .stSlider * {
    color: #fff !important;
    font-size: 1.08rem;
}

.result-box {
    margin: 30px auto;
    border-radius: 15px;
    box-shadow: 0 6px 15px rgba(0,0,0,0.15);
    text-align: center;
    padding: 24px;
    font-size: 1.5rem;
    max-width: 800px;
    color: #fff;
}
.result-box.no { background: rgba(46, 204, 113, 0.85); }
.result-box.low { background: rgba(255, 196, 12, 0.85); color: #222; }
.result-box.medium { background: rgba(255, 165, 0, 0.85); color: #222; }
.result-box.high { background: rgba(231, 76, 60, 0.85); }

.stForm button[kind="formSubmit"], .stForm .stButton button {
    background-color: #fff !important;
    color: #333 !important;
    border: 2px solid #ddd !important;
    border-radius: 8px !important;
    font-size: 16px !important;
    padding: 10px 25px !important;
    font-weight: 600 !important;
}
.stForm button[kind="formSubmit"]:hover, .stForm .stButton button:hover {
    background-color: #2980b9 !important;
    color: white !important;
    border-color: #2980b9 !important;
}

h1 {
    font-size: 2.5rem;
    font-weight: bold;
    color: #2c3e50;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("💉 Diabetes Risk Prediction")

# Helper functions
def create_radio(label, key):
    # Create radio button input for yes/no options, returns the selected value (0 for "No", 1 for "Yes")
    return st.radio(label, [("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key=key)[1]

def get_risk_level(prob):
    # Determines the risk level based on the predicted probability (prob)
    for (min_val, max_val), (level, css_class) in RISK_LEVELS.items():
        if min_val <= prob < max_val:
            return level, css_class
    return "High Risk", "high"  # Default fallback if not in any category

# Form
with st.form("diabetes_form"):
    form_data = {
        'HighBP': create_radio("Do you have high blood pressure? (More than 130/80 mmHg)", "bp"),
        'HighChol': create_radio("Do you have high cholesterol? (More than 240 mg/dL)", "chol"),
        'BMI': st.slider("BMI", 10, 60, 24),
        'Stroke': create_radio("Have you ever had a stroke?", "stroke"),
        'HeartDiseaseorAttack': create_radio("Have you had heart disease or heart attack?", "heart"),
        'PhysActivity': create_radio("Do you do any physical activities?", "activity"),
        'HvyAlcoholConsump': create_radio("Heavy alcohol consumption? (More than 4 drinks a day for men, 3 for women)", "alcohol"),
        'GenHlth': st.selectbox("General Health (1=Excellent, 5=Poor)", range(1, 6)),
        'DiffWalk': create_radio("Do you have difficulty walking?", "walk"),
        'Sex': st.radio("Gender", [("Female", 0), ("Male", 1)], format_func=lambda x: x[0])[1],
        'Age': st.slider("Age", 0, 100, 18)
    }
    
    # Wait for the form submission
    submitted = st.form_submit_button("Submit")

if submitted:
    # Preparing the form input as a dataframe for prediction
    input_data = [form_data[feature] for feature in MODEL_FEATURES]
    input_df = pd.DataFrame([input_data], columns=MODEL_FEATURES)
    
    # Model prediction
    prob = model.predict_proba(input_df)[0, 1]  # Probability of being diabetic
    is_diabetic = prob >= threshold  # Check if the probability exceeds the threshold
    
    # Display results based on prediction
    if is_diabetic:
        risk_level, box_class = get_risk_level(prob)  # Determine the risk level
        result_text = f"Our model predicts you are <strong>Diabetic</strong>. Risk Level: <strong>{risk_level}</strong>."
    else:
        box_class = "no"
        result_text = "Our model predicts you are <strong>Not Diabetic</strong>."
    
    st.markdown(f'<div class="result-box {box_class}">{result_text}</div>', unsafe_allow_html=True)

