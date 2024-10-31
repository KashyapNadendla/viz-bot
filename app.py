# app.py

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os

# Import functions from other files
from eda import eda_section
from data_cleaning import data_cleaning_section
from ml_modeling import ml_modeling_section
from visualizations import visualization_section
from llm_analysis import llm_analysis_section

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Data the Explorer", page_icon="📈")

st.title("Data the Explorer")
st.caption("Data Analysis & LLM Integration")

# Brief Explanation
st.write("""
Data the Explorer is an interactive application tailored for comprehensive data exploration, EDA, and machine learning model evaluation. Users can upload datasets to perform detailed Exploratory Data Analysis (EDA), build and run ML models, and receive interpretations of model results. Enhanced by Large Language Models (LLMs), the app provides AI-driven recommendations on visualizations, feature engineering, and model choices, making data analysis more insightful and accessible.
""")

# File uploader widget to allow users to upload a CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    # Get the uploaded file's details
    file_details = {"filename": uploaded_file.name, "filesize": uploaded_file.size}

    # Check if this is a new file or if data needs to be updated
    if (
        'file_details' not in st.session_state
        or st.session_state.file_details != file_details
    ):
        # Read the CSV file
        data = pd.read_csv(uploaded_file)

        # Store data and file details in session state
        st.session_state.original_data = data.copy()  # Store the original data
        st.session_state.data = data.copy()  # Working copy of the data
        st.session_state.file_details = file_details  # Store file details
        st.success("Data has been updated with the new file.")
    else:
        data = st.session_state.data  # Use the data from session_state

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        [
            "EDA",
            "Data Cleaning and Preprocessing",
            "Machine Learning Modeling and Evaluation",
            "Interactive Visualizations",
            "LLM Analysis and Suggestions"
        ]
    )

    if app_mode == "EDA":
        eda_section()
    elif app_mode == "Data Cleaning and Preprocessing":
        data_cleaning_section()
    elif app_mode == "Machine Learning Modeling and Evaluation":
        ml_modeling_section()
    elif app_mode == "Interactive Visualizations":
        visualization_section()
    elif app_mode == "LLM Analysis and Suggestions":
        llm_analysis_section()
else:
    st.write("Please upload a CSV file to proceed.")
