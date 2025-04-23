import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
try:
    with open("C:/Users/Mariam/Downloads/rf_model.pkl", "rb") as f:
      model = pickle.load(f)

    with open("C:/Users/Mariam/Downloads/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError as e:
    st.error(f"Error: {e}. Please ensure rf_model.pkl and scaler.pkl are in C:/project/notebook/")
    st.stop()

# Define the top features used in the model (from correlation analysis)
top_features = [
    'Income composition of resources', 'Schooling', 'Adult Mortality', 
    ' HIV/AIDS', ' BMI ', ' thinness  1-19 years', 'thinness 5-9 years', 
    'Diphtheria ', 'Polio', 'Total expenditure', 'GDP', 'Hepatitis B', 
    'Alcohol', 'Status_Developed', 'under-five deaths '
]

# Streamlit app
st.title("Life Expectancy Prediction")
st.write("Enter the values for the following features to predict life expectancy.")

# Create input fields for each feature
input_data = {}
with st.form("prediction_form"):
    for feature in top_features:
        if feature == 'Status_Developed':
            # Binary input for Status_Developed (0 or 1)
            input_data[feature] = st.selectbox(feature, options=[0, 1], format_func=lambda x: "Developed" if x == 1 else "Developing")
        else:
            # Numeric input for other features
            input_data[feature] = st.number_input(feature, step=0.1, format="%.2f")

    # Submit button
    submitted = st.form_submit_button("Predict")

# Process prediction when form is submitted
if submitted:
    try:
        # Create a DataFrame from input data
        input_df = pd.DataFrame([input_data], columns=top_features)
        
        # Scale the input data
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Display the result
        st.success(f"Predicted Life Expectancy: {prediction:.2f} years")
        
        # Optional: Display input values for verification
        st.write("Input Values:")
        st.json(input_data)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Add some information about the model
st.markdown("""
### About the Model
This app uses a Random Forest Regression model trained on the Life Expectancy dataset.
The model predicts life expectancy based on the top 15 correlated features.
Ensure all inputs are provided accurately for the best prediction results.
""")