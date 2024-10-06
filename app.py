import streamlit as st
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI 
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Function to convert the dataset into JSON format
def convert_data_to_json(data):
    """
    This function converts the uploaded dataset into JSON format.
    The dataset is converted into 'records' format to create a list of dictionaries, 
    with each row in the dataset represented as a dictionary.

    Returns:
        JSON formatted string of the dataset.
    """
    json_data = data.to_dict(orient='records')  # Convert to JSON records format
    return json.dumps(json_data, indent=2)

# Function to interact with the OpenAI API for analyzing data, suggesting models, or suggesting visualizations
def analyze_data_with_llm(json_data, prompt_type="analysis"):
    """
    This function sends the dataset (in JSON format) to the OpenAI model using LangChain.
    Based on the specified prompt_type, it generates a relevant prompt for either data analysis,
    model suggestions, or visualization suggestions, and retrieves the LLM's response.

    Parameters:
        json_data (str): JSON representation of the dataset.
        prompt_type (str): The type of prompt - "analysis", "model_suggestions", or "visualization_suggestions".

    Returns:
        A response string from the LLM with analysis, model suggestions, or visualization ideas.
    """
    # Initialize the LLM model
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini")

    # Generate a prompt based on the request type
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

    # Run the chain with the JSON dataset and return the response
    response = chain.run({"json_data": json_data})
    
    return response.strip()

# Streamlit app user interface
st.title("Data Analysis and EDA with LLM Integration")

# File uploader widget to allow users to upload a CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    # Read the CSV file using pandas
    data = pd.read_csv(uploaded_file)

    # Display the uploaded data in the Streamlit app
    st.write("## Uploaded Data")
    st.dataframe(data)
    
    # Display the number of rows in the dataset
    num_rows = data.shape[0]
    st.write(f"**Number of rows in the dataset:** {num_rows}")
    
    # Check if the dataset has more than 10,000 rows and provide an option to limit it to 500 rows
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

    # Convert the dataset to JSON format for analysis
    json_data = convert_data_to_json(data)

    # Exploratory Data Analysis (EDA)
    st.header("Exploratory Data Analysis")

    # Show basic statistics
    if st.checkbox("Show basic statistics"):
        st.write("### Basic Statistics")
        st.write(data.describe())

    # Correlation Matrix
    if st.checkbox("Show correlation matrix"):
        st.write("### Correlation Matrix")
        corr_matrix = data.corr()
        st.write(corr_matrix)

        # Heatmap
        st.write("#### Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    # Pairplot
    if st.checkbox("Show pairplot"):
        st.write("### Pairplot")
        fig = sns.pairplot(data)
        st.pyplot(fig)

    # Histograms
    if st.checkbox("Show histograms"):
        st.write("### Histograms")
        numeric_columns = data.select_dtypes(include=['float', 'int']).columns
        for col in numeric_columns:
            st.write(f"#### Histogram for {col}")
            fig, ax = plt.subplots()
            sns.histplot(data[col], kde=True, ax=ax)
            st.pyplot(fig)

    # Scatter plots
    if st.checkbox("Show scatter plots"):
        st.write("### Scatter Plots")
        numeric_columns = data.select_dtypes(include=['float', 'int']).columns
        if len(numeric_columns) >= 2:
            x_axis = st.selectbox("Select X-axis variable", options=numeric_columns)
            y_axis = st.selectbox("Select Y-axis variable", options=numeric_columns)
            if x_axis and y_axis:
                st.write(f"#### Scatter plot between {x_axis} and {y_axis}")
                fig, ax = plt.subplots()
                sns.scatterplot(x=data[x_axis], y=data[y_axis], ax=ax)
                st.pyplot(fig)
        else:
            st.write("Not enough numerical columns for scatter plot.")

    # Button to trigger LLM analysis of the dataset
    st.header("LLM Analysis and Suggestions")
    if st.button("Analyze Data"):
        analysis_response = analyze_data_with_llm(json_data, prompt_type="analysis")
        
        # Display the analysis and suggestions from the LLM
        st.write("### Data Analysis and Improvement Suggestions:")
        st.write(analysis_response)

    # Button to trigger LLM model suggestions based on the dataset
    if st.button("Suggest Models"):
        model_suggestions_response = analyze_data_with_llm(json_data, prompt_type="model_suggestions")
        
        # Display the model suggestions from the LLM
        st.write("### Model Suggestions:")
        st.write(model_suggestions_response)
    
    # Button to trigger LLM visualization suggestions based on the dataset
    if st.button("Suggest Visualizations"):
        visualization_suggestions_response = analyze_data_with_llm(json_data, prompt_type="visualization_suggestions")
        
        # Display the visualization suggestions from the LLM
        st.write("### Visualization Suggestions:")
        st.write(visualization_suggestions_response)

else:
    # Display a message to prompt the user to upload a file
    st.write("Please upload a CSV file to proceed.")
