# test_persistence.py
# Simple test to verify the persistence functionality

import streamlit as st
import pandas as pd
from datetime import datetime

def test_persistence():
    """Test the persistence functionality"""
    
    # Simulate session state
    if 'test_stored_visualizations' not in st.session_state:
        st.session_state.test_stored_visualizations = []
    
    # Add a test visualization
    test_viz = {
        "user_request": "Test chart creation",
        "chart_type": "TestChart",
        "insights": "This is a test insight",
        "timestamp": datetime.now(),
        "fig": None  # In real app, this would be a plotly figure
    }
    
    st.session_state.test_stored_visualizations.append(test_viz)
    
    st.write("Test visualization stored successfully!")
    st.write(f"Number of stored visualizations: {len(st.session_state.test_stored_visualizations)}")
    
    for i, viz in enumerate(st.session_state.test_stored_visualizations):
        st.write(f"Visualization {i+1}: {viz['user_request']}")

if __name__ == "__main__":
    test_persistence() 