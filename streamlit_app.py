import streamlit as st
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt

# Load your trained model
# You need to make sure the model is in the same directory as this script
# or provide the correct path to the .pkl file
# It's also assumed that you've pickled your trained model into a .pkl file
@st.cache
def load_model():
    import joblib
    model = joblib.load('finalized_model.pkl') 
    return model

# This function makes the forecast using the provided model and inputs
def make_forecast(model, inputs):
    forecast = model.predict([inputs])
    return forecast

model = load_model()

st.title('Surgery Time Forecasting System')

# Input cells for user input
input1 = st.number_input('Patient age (0 - 97)', value=0.0)
input2 = st.number_input('Patient gender (0->F,  1->M)', value=0.0)
input3 = st.number_input('Surgery (4 - 2738)', value=0.0)
input4 = st.number_input('Surgery preparation time (5 - 990)', value=0.0)
input5 = st.number_input('Anesthesia (0 - 78)', value=0.0)


inputs = [input1, input2, input3, input4, input5]

# Make prediction and get confidence interval
if st.button('Make a forecast'):
    forecast = make_forecast(model, inputs)
    st.write(f'Expected Average Time: {forecast[0]} minutes')
    

    # Assuming the model can return prediction intervals
    # This may need to be adjusted based on your model and data
    lower_bound = forecast[0] - 0.1 * forecast[0]
    upper_bound = forecast[0] + 0.1 * forecast[0]
    
    st.write(f'Minimum time required: {lower_bound} minutes')
    st.write(f'Maximum time required: {upper_bound} minutes')

#     fig, ax = plt.subplots()
#     ax.plot(forecast[0], color='blue', label='Predicted')
#     ax.fill_between(np.arange(len(forecast)), lower_bound, upper_bound, color='gray', alpha=0.6, label='Confidence Interval')
#     ax.legend()

#     st.pyplot(fig)
