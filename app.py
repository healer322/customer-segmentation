import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FFFFFF;
        padding: 1rem 0;
        text-align: center;
        background: linear-gradient(90deg, #1E3A8A, #2563EB);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #3B82F6;
        padding: 0.5rem 0;
        border-bottom: 2px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #1E293B;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .prediction-result {
        font-size: 1.8rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model_path = "customer_segmentation_model1 (1).pkl"
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Load the trained model
model = load_model()

# Main header
st.markdown('<div class="main-header"> Customer Segmentation Predictor</div>', unsafe_allow_html=True)

# Create input columns
col1, col2 = st.columns(2)
with col1:
    st.subheader("Customer Information")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    annual_income = st.number_input("Annual Income (k$)", min_value=0, value=50)

with col2:
    st.subheader("Shopping Behavior")
    spending_score = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=50)

# Center the predict button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button('Predict Customer Segment', use_container_width=True)

if predict_button:
    # Prepare input data
    input_data = np.array([[annual_income, spending_score]])
    
    # Predict cluster
    cluster = model.predict(input_data)[0]
    
    # Display results
    st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)

    with st.container():
        st.success(f"✅ Customer belongs to Segment {cluster}")

        # Display input details
        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.write("📊 Customer Profile")
            st.write(f"Age: {age}")
            st.write(f"Annual Income: {annual_income}k$")
            st.write(f"Spending Score: {spending_score}")
            st.markdown('</div>', unsafe_allow_html=True)


            # Sample customer inputs and predictions
st.markdown('<div class="section-header">Sample Customer Profiles</div>', unsafe_allow_html=True)

example_data = {
    "Profile": ["High Spender", "Low Spender", "Average Spender"],
    "Age": [35, 50, 28],
    "Annual Income (k$)": [75, 30, 55],
    "Spending Score": [85, 20, 50]
}

example_df = pd.DataFrame(example_data)

# Predict segments for sample customers
example_input = example_df[["Annual Income (k$)", "Spending Score"]].values
predicted_clusters = model.predict(example_input)
example_df["Predicted Segment"] = predicted_clusters


#st.write("\nExample customer inputs:")
#st.write("High spender: Age 35, Income 75k$, Spending Score 85")
#st.write("Low spender: Age 50, Income 30k$, Spending Score 20")
