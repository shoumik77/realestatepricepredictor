import streamlit as st
import numpy as np
import joblib

# Load your scaler and model
scaler = joblib.load("Scaler.pkl")
model = joblib.load("model.pkl")



st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        

        /* Center-align everything */
        .main {
            text-align: center;
        }

        /* Customize input widgets */
        input, select {
            background-color: #ffffff;
            color: #000000;
        }

        /* Enhance button appearance */
        button {
            background-color: #4caf50;
            color: white;
            font-weight: bold;
            border-radius: 5px;
        }

        

        h2, h3 {
            color: #555555;
        }

        /* Prediction box styling */
        .prediction-box {
            background-color: #ffffff;
            border: 2px solid #dddddd;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            font-size: 24px;
            color: #333333;
            font-weight: bold;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Title
st.title("🏠 Real Estate Price Predictor")

st.divider()

# Input fields
bed = st.number_input("Enter Number of Bedrooms", value=2, step=1)
bath = st.number_input("Enter Number of Bathrooms", value=1, step=1)
size = st.number_input("Enter the Size (in sqft)", value=1000, step=50)

x = [bed, bath, size]

st.divider()

# Predict button
predictbutton = st.button("✨ Predict Price")

st.divider()

# Display prediction result
if predictbutton:
    x1 = np.array(x)
    x_array = scaler.transform([x1])
    prediction = model.predict(x_array)[0]  # Extract the single value
    
    st.markdown(
        f"""
        <div class="prediction-box">
            🏡 Predicted Price: <span style="color: #4caf50;">${float(prediction):,.2f}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div style="
            text-align: center;
            font-size: 18px;
            color: #555555;
            margin-top: 20px;
        ">
            Please click the "✨ Predict Price" button above to get the estimated price.
        </div>
        """,
        unsafe_allow_html=True,
    )
