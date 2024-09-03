import streamlit as st
import pandas as pd
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Your OpenAI API key

# Function to perform statistical analysis on data
def perform_statistical_analysis(data, selected_columns):
    analysis_results = {}
    
    for column in selected_columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            # Perform numerical analysis
            stats = {
                "mean": data[column].mean(),
                "median": data[column].median(),
                "std_dev": data[column].std(),
                "min": data[column].min(),
                "max": data[column].max(),
                "quartiles": data[column].quantile([0.25, 0.5, 0.75]).to_dict()
            }
        elif pd.api.types.is_categorical_dtype(data[column]) or pd.api.types.is_object_dtype(data[column]):
            # Perform categorical analysis
            stats = {
                "unique_values": data[column].nunique(),
                "most_frequent": data[column].mode()[0] if not data[column].mode().empty else None,
                "value_counts": data[column].value_counts().to_dict()
            }
        elif pd.api.types.is_datetime64_any_dtype(data[column]):
            # Perform datetime analysis
            stats = {
                "min_date": data[column].min(),
                "max_date": data[column].max(),
                "frequency": pd.infer_freq(data[column].dropna()) if not data[column].dropna().empty else None
            }
        else:
            # Unsupported data type
            stats = {}

        # Add stats to the results
        analysis_results[column] = {
            "data_type": str(data[column].dtype),
            "statistics": stats
        }
    
    return analysis_results

# Function to generate prompt for LLM with selected columns
def generate_llm_prompt_for_selected_columns(analysis_results):
    prompt = "Based on the following statistics, suggest the best type of visualization and provide the Python code to create it using matplotlib:\n"
    
    for column, info in analysis_results.items():
        prompt += f"\nColumn '{column}' ({info['data_type']}):\n"
        for stat_name, stat_value in info["statistics"].items():
            # Safely access the quartile values if they exist
            if stat_name == "quartiles":
                quartiles = info["statistics"]["quartiles"]
                for quartile, value in quartiles.items():
                    prompt += f"  - {quartile}: {value}\n"
            else:
                prompt += f"  - {stat_name}: {stat_value}\n"
    
    prompt += "\nBased on this information, suggest the most suitable visualization types for these columns and provide the Python code to create them using matplotlib."
    return prompt

# Function to get visualization suggestions using LangChain
def get_visualization_suggestions(prompt, api_key):
    # Initialize ChatOpenAI with the API key and model
    llm = ChatOpenAI(openai_api_key=api_key, model="gpt-4o-mini")

    # Create a chat prompt template using LangChain
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt),
            ("human", "What visualization would be most appropriate for the selected columns, and provide the Python code to create it using matplotlib.")
        ]
    )

    # Create a chain using LangChain's LLMChain
    chain = LLMChain(llm=llm, prompt=chat_prompt)
    response = chain.run({"input": ""})  # No input is needed since the prompt contains all necessary information
    
    return response.strip()

# Function to parse LLM's response and extract Python code
def parse_llm_response_for_code(llm_response):
    # Simplistic parsing logic assuming the response format is consistent
    code_start = llm_response.find("```python")
    code_end = llm_response.find("```", code_start + 1)
    if code_start != -1 and code_end != -1:
        return llm_response[code_start + len("```python"):code_end].strip()
    return None

# Streamlit app
st.title("Dynamic CSV Visualization Suggestion")

# File uploader widget
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)

    # Display the uploaded data
    st.write("Uploaded Data:")
    st.dataframe(data)

    # Column selection for visualization
    st.write("Select Two Columns for Visualization:")
    selected_columns = st.multiselect("Select columns", data.columns.tolist(), default=data.columns.tolist()[:2])

    if len(selected_columns) == 2:
        # Perform statistical analysis for selected columns
        analysis_results = perform_statistical_analysis(data, selected_columns)
        
        # Display statistical analysis for selected columns
        st.write("Statistical Analysis for Selected Columns:")
        for column, stats in analysis_results.items():
            st.write(f"**{column}** ({stats['data_type']}):")
            for stat_name, stat_value in stats["statistics"].items():
                if stat_name == "quartiles":
                    st.write("- quartiles:")
                    for quartile, value in stat_value.items():
                        st.write(f"  - {quartile}: {value}")
                else:
                    st.write(f"- {stat_name}: {stat_value}")
        
        # Generate and display LLM prompt with selected columns
        llm_prompt = generate_llm_prompt_for_selected_columns(analysis_results)
        st.write("Generated Prompt for LLM:")
        st.code(llm_prompt)

        # Button to get visualization suggestions
        if st.button("Get Visualization Suggestions"):
            # Get visualization suggestions from LLM
            visualization_suggestions = get_visualization_suggestions(llm_prompt, OPENAI_API_KEY)
            st.write("Visualization Suggestions and Code from LLM:")
            st.write(visualization_suggestions)

            # Parse the LLM response to extract Python code
            python_code = parse_llm_response_for_code(visualization_suggestions)

            if python_code:
                st.write("Executing the following Python code to generate the visualization:")
                st.code(python_code)

                # Safely execute the extracted Python code
                try:
                    exec(python_code)
                    st.pyplot(plt)  # Display the plot generated by matplotlib
                except Exception as e:
                    st.error(f"An error occurred while executing the code: {e}")
    else:
        st.write("Please select exactly two columns for visualization.")
else:
    st.write("Please upload a CSV file to see its column information and statistical analysis.")
