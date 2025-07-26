# app.py

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os

# --- Page Configuration ---
# Set page config must be the first Streamlit command
st.set_page_config(page_title="Data the Explorer", page_icon="📈", layout="wide")

# --- Import Section Functions ---
# These functions are defined in their own respective files
from eda import eda_section
from data_cleaning import data_cleaning_section
from ml_modeling import ml_modeling_section
from unified_visualizations import unified_visualization_section
from llm_analysis import llm_analysis_section
from data_mining_engine import data_mining_engine_section # New import

# --- Load Environment Variables ---
load_dotenv()

# --- Main App Logic ---

def structured_data_app():
    """
    Handles all functionality related to structured data analysis (CSVs).
    """
    # File uploader for CSV
    uploaded_file = st.file_uploader("Upload a CSV file to explore", type="csv", key="structured_uploader")

    if uploaded_file:
        # Check if this is a new file or if data needs to be re-read
        if ('file_details' not in st.session_state or 
            st.session_state.file_details["filename"] != uploaded_file.name or
            st.session_state.file_details["filesize"] != uploaded_file.size):
            
            data = pd.read_csv(uploaded_file)
            st.session_state.original_data = data.copy()
            st.session_state.data = data.copy()
            st.session_state.file_details = {"filename": uploaded_file.name, "filesize": uploaded_file.size}
            st.success("New dataset loaded successfully.")
        
        # --- Secondary Navigation for Structured Data ---
        st.sidebar.subheader("Structured Data Tools")
        app_mode = st.sidebar.selectbox(
            "Choose an analysis tool",
            [
                "EDA",
                "Data Cleaning and Preprocessing",
                "Machine Learning Modeling and Evaluation",
                "🤖 Unified AI Visualization Assistant",
                "LLM Analysis and Suggestions"
            ]
        )

        # --- Call the appropriate section function ---
        if app_mode == "EDA":
            eda_section()
        elif app_mode == "Data Cleaning and Preprocessing":
            data_cleaning_section()
        elif app_mode == "Machine Learning Modeling and Evaluation":
            ml_modeling_section()
        elif app_mode == "🤖 Unified AI Visualization Assistant":
            unified_visualization_section()
        elif app_mode == "LLM Analysis and Suggestions":
            llm_analysis_section()
    else:
        st.info("Please upload a CSV file to access the structured data analysis tools.")


def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("Data the Explorer")
    st.caption("An AI-powered toolkit for structured and unstructured data analysis.")

    # Keep the original introduction on the main page
    st.write("""
    Data the Explorer is an interactive application tailored for comprehensive data exploration, EDA, and machine learning model evaluation. Users can upload datasets to perform detailed Exploratory Data Analysis (EDA), build and run ML models, and receive interpretations of model results. Enhanced by Large Language Models (LLMs), the app provides AI-driven recommendations on visualizations, feature engineering, and model choices, making data analysis more insightful and accessible.
    """)
    st.write("---")

    # --- Primary Navigation ---
    st.sidebar.title("Navigation")
    main_mode = st.sidebar.radio(
        "Select an Analysis Mode",
        ("Structured Data (CSV)", "Unstructured Data (PDFs)")
    )

    if main_mode == "Structured Data (CSV)":
        structured_data_app()
    elif main_mode == "Unstructured Data (PDFs)":
        data_mining_engine_section()

if __name__ == "__main__":
    main()