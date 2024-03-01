import streamlit as st
import pandas as pd
from joblib import load


# command to run the app
# streamlit run app.py

# method to load the model
def load_model():
    return load("/Users/mikkel/Documents/GitHub/MP4/decision_tree_model.joblib")

# method to predict attrition
def predict_attrition(model, input_data):
    # map input values so they match with the model
    input_data['OverTime'] = input_data['OverTime'].map({'Yes': 1, 'No': 0})
    input_data['BusinessTravel'] = input_data['BusinessTravel'].map({'Travel_Rarely': 1, 'Travel_Frequently': 2, 'Non-Travel': 0})
    input_data['MaritalStatus'] = input_data['MaritalStatus'].map({'Single': 1, 'Married': 2, 'Divorced': 0})

    prediction = model.predict(input_data)
    return prediction

# load the model
model = load_model()

st.set_page_config(
    page_title="Attrition Prediction",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:tdi@cphbusiness.dk',
        'About': "https://docs.streamlit.io"
    }
)

st.title("Attrition Prediction")

# formular for input 
st.sidebar.header("Input Data")
overtime = st.sidebar.selectbox("OverTime", ["Yes", "No"])
business_travel = st.sidebar.selectbox("BusinessTravel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
total_working_years = st.sidebar.number_input("Total Working Years", min_value=0, value=5)
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
years_in_current_role = st.sidebar.number_input("Years In Current Role", min_value=0, value=2)

# colelct input data as a dataframe
input_data = pd.DataFrame({
    'OverTime': [overtime],
    'BusinessTravel': [business_travel],
    'TotalWorkingYears': [total_working_years],
    'MaritalStatus': [marital_status],
    'YearsInCurrentRole': [years_in_current_role]
})

# predict attrition based on input data
prediction = predict_attrition(model, input_data)

# show result
st.header("Attrition Prediction Result")
st.write("Predicted Attrition:", "Yes" if prediction[0] == 1 else "No")
