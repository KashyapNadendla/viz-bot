import streamlit as st
import pandas as pd
import json
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI  # Corrected import
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from environment variables


# Function to convert the data into JSON format
def convert_data_to_json(data):
    # Create a summary of the dataset in JSON format
    json_data = data.to_dict(orient='records')  # Convert to JSON records format
    return json.dumps(json_data, indent=2)

# Function to send the JSON data to OpenAI using LangChain for analysis
def analyze_data_with_llm(json_data, prompt_type="analysis"):
    # Initialize the LLM model
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini")

    # Generate the appropriate prompt based on the request type
    if prompt_type == "analysis":
        prompt_template = PromptTemplate(
            input_variables=["json_data"],
            template="Here is a dataset represented in JSON format:\n{json_data}\n\nPlease analyze the dataset, identify any issues such as missing values or inconsistencies, and provide suggestions for improvement."
        )
    elif prompt_type == "model_suggestions":
        prompt_template = PromptTemplate(
            input_variables=["json_data"],
            template="Here is a dataset represented in JSON format:\n{json_data}\n\nBased on this dataset, please suggest the best machine learning models for analysis, considering the types of columns and their contents."
        )
    
    elif prompt_type == "visualization_suggestions":
        prompt_template = PromptTemplate(
            input_variables=["json_data"],
            template="Here is a dataset represented in JSON format:\n{json_data}\n\nBased on the structure and contents of this dataset, please suggest the best visualizations that can help explore and understand the data."
        )

    # Create the LLMChain with the prompt template
    chain = LLMChain(llm=llm, prompt=prompt_template)

    # Run the chain, passing in the JSON data as the 'json_data' variable
    response = chain.run({"json_data": json_data})
    
    return response.strip()

# Streamlit app
st.title("Send Data to LLM in JSON Format Using LangChain")

# File uploader widget
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)

    # Display the uploaded data
    st.write("Uploaded Data:")
    st.dataframe(data)
    
    # Get the number of rows
    num_rows = data.shape[0]
    st.write(f"Number of rows in the dataset: {num_rows}")
    
    # If more than 10,000 rows, give an option to use only the first 500 rows
    if num_rows > 10000:
        use_limited_data = st.checkbox("The dataset exceeds 10,000 rows. Check this box to use only the first 500 rows.")
        if use_limited_data:
            data = data.head(500)
            st.write("Using the first 500 rows of the dataset.")
        else:
            st.warning("The dataset is too large to process. Please choose to use the first 500 rows or upload a smaller dataset.")
            st.stop()  # Stop the app from proceeding if no option is chosen
    else:
        st.write("The dataset has fewer than 10,000 rows.")

    # Convert the dataset to JSON format
    json_data = convert_data_to_json(data)

    # Button to send the JSON data to the LLM for analysis
    if st.button("Analyze Data"):
        # Send the JSON data to the LLM using LangChain
        analysis_response = analyze_data_with_llm(json_data, prompt_type="analysis")
        
        # Display the analysis from the LLM
        st.write("LLM Analysis and Suggestions:")
        st.write(analysis_response)

    # Button to send the JSON data to the LLM for model suggestions
    if st.button("Suggest Models"):
        # Send the JSON data to the LLM using LangChain for model suggestions
        model_suggestions_response = analyze_data_with_llm(json_data, prompt_type="model_suggestions")
        
        # Display the model suggestions from the LLM
        st.write("Model Suggestions from LLM:")
        st.write(model_suggestions_response)
    
     # Button to send the JSON data to the LLM for visualization suggestions
    if st.button("Suggest Visualizations"):
        # Send the JSON data to the LLM using LangChain for visualization suggestions
        visualization_suggestions_response = analyze_data_with_llm(json_data, prompt_type="visualization_suggestions")
        
        # Display the visualization suggestions from the LLM
        st.write("Visualization Suggestions from LLM:")
        st.write(visualization_suggestions_response)

else:
    st.write("Please upload a CSV file to proceed.")
