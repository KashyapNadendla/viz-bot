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
    Converts the uploaded dataset into JSON format.
    The dataset is converted into 'records' format to create a list of dictionaries,
    with each row in the dataset represented as a dictionary.

    Returns:
        JSON formatted string of the dataset.
    """
    json_data = data.to_dict(orient='records')  # Convert to JSON records format
    return json.dumps(json_data, indent=2)

# Function to interact with the OpenAI API for analyzing data, suggesting models, visualization suggestions, or feature engineering
def analyze_data_with_llm(json_data, prompt_type="analysis", user_question=None):
    """
    Sends the dataset (in JSON format) to the OpenAI model using LangChain.
    Based on the specified prompt_type, it generates a relevant prompt for data analysis,
    model suggestions, visualization suggestions, or feature engineering, and retrieves the LLM's response.

    Parameters:
        json_data (str): JSON representation of the dataset.
        prompt_type (str): The type of prompt - "analysis", "model_suggestions", "visualization_suggestions", "feature_engineering", or "custom".
        user_question (str): Custom user question for the LLM.

    Returns:
        A response string from the LLM with analysis, suggestions, or answers.
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
    elif prompt_type == "feature_engineering":
        prompt_template = PromptTemplate(
            input_variables=["json_data"],
            template="Here is a dataset represented in JSON format:\n{json_data}\n\nPlease suggest potential feature engineering techniques that could improve model performance."
        )
    elif prompt_type == "custom" and user_question:
        prompt_template = PromptTemplate(
            input_variables=["json_data", "user_question"],
            template="Here is a dataset represented in JSON format:\n{json_data}\n\nQuestion: {user_question}\nAnswer:"
        )
    else:
        return "Invalid prompt type or missing user question."

    # Create the LLMChain with the prompt template
    chain = LLMChain(llm=llm, prompt=prompt_template)

    # Run the chain with the JSON dataset and return the response
    if prompt_type == "custom":
        response = chain.run({"json_data": json_data, "user_question": user_question})
    else:
        response = chain.run({"json_data": json_data})
    
    return response.strip()

# Streamlit app user interface
st.title("Data Analysis and EDA with LLM Integration")

# File uploader widget to allow users to upload a CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    # Read the CSV file using pandas
    data = pd.read_csv(uploaded_file)

    # Convert date columns to datetime
    for col in data.columns:
        if 'date' in col.lower():
            data[col] = pd.to_datetime(data[col], errors='coerce')

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

    # Group related features under expandable sections
    with st.expander("Data Overview and Summary Statistics"):
        # Data Shape and Types
        if st.checkbox("Show Data Shape and Types", help="Displays the shape and data types of the dataset."):
            st.write(f"Data has {data.shape[0]} rows and {data.shape[1]} columns.")
            st.write("Data Types:")
            st.write(data.dtypes)

        # Missing Values
        if st.checkbox("Show Missing Values", help="Displays the count of missing values in each column."):
            st.write("### Missing Values")
            st.write(data.isnull().sum())

        # Descriptive Statistics
        if st.checkbox("Show Descriptive Statistics", help="Displays basic statistical measures for numerical columns."):
            st.write("### Descriptive Statistics")
            st.write(data.describe())

    with st.expander("Data Distribution and Visualization"):
        # Histograms and Density Plots
        if st.checkbox("Show Histograms", help="Displays histograms for numerical variables to visualize distributions."):
            st.write("### Histograms")
            numeric_cols = data.select_dtypes(include=['float', 'int']).columns
            for col in numeric_cols:
                st.write(f"#### Histogram for {col}")
                fig, ax = plt.subplots()
                try:
                    sns.histplot(data[col], kde=True, ax=ax)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Could not plot histogram for {col}: {e}")

        # Box Plots
        if st.checkbox("Show Box Plots", help="Displays box plots for numerical variables to detect outliers."):
            st.write("### Box Plots")
            numeric_cols = data.select_dtypes(include=['float', 'int']).columns
            for col in numeric_cols:
                st.write(f"#### Box Plot for {col}")
                fig, ax = plt.subplots()
                try:
                    sns.boxplot(y=data[col], ax=ax)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Could not plot box plot for {col}: {e}")

        # Violin Plots
        if st.checkbox("Show Violin Plots", help="Displays violin plots for numerical variables to visualize distributions."):
            st.write("### Violin Plots")
            numeric_cols = data.select_dtypes(include=['float', 'int']).columns
            for col in numeric_cols:
                st.write(f"#### Violin Plot for {col}")
                fig, ax = plt.subplots()
                try:
                    sns.violinplot(y=data[col], ax=ax)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Could not plot violin plot for {col}: {e}")

        # Scatter Plots
        if st.checkbox("Show Scatter Plots", help="Displays scatter plots for selected numerical variables."):
            st.write("### Scatter Plots")
            numeric_cols = data.select_dtypes(include=['float', 'int']).columns
            if len(numeric_cols) >= 2:
                x_axis = st.selectbox("Select X-axis variable", options=numeric_cols, key='scatter_x')
                y_axis = st.selectbox("Select Y-axis variable", options=numeric_cols, key='scatter_y')
                if x_axis and y_axis:
                    st.write(f"#### Scatter Plot between {x_axis} and {y_axis}")
                    fig, ax = plt.subplots()
                    try:
                        sns.scatterplot(x=data[x_axis], y=data[y_axis], ax=ax)
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Could not generate scatter plot: {e}")
            else:
                st.write("Not enough numerical columns to create a scatter plot.")

    with st.expander("Correlation and Relationships"):
        # Correlation Matrix and Heatmap
        if st.checkbox("Show Correlation Heatmap", help="Displays a heatmap of the correlation matrix for numerical variables."):
            st.write("### Correlation Heatmap")
            numeric_cols = data.select_dtypes(include=['float', 'int']).columns
            if len(numeric_cols) >= 2:
                corr = data[numeric_cols].corr()
                fig, ax = plt.subplots()
                try:
                    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Could not generate correlation heatmap: {e}")
            else:
                st.write("Not enough numerical columns to compute correlation.")

        # Pair Plot
        if st.checkbox("Show Pair Plot", help="Displays pairwise relationships between variables."):
            st.write("### Pair Plot")
            try:
                fig = sns.pairplot(data)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Could not generate pair plot: {e}")

    with st.expander("Outlier Detection"):
        # Outlier Detection using Z-Score
        if st.checkbox("Detect Outliers (Z-Score)", help="Identifies outliers in numerical variables using Z-Score method."):
            st.write("### Outlier Detection using Z-Score")
            from scipy import stats
            numeric_cols = data.select_dtypes(include=['float', 'int']).columns
            outliers = {}
            for col in numeric_cols:
                z_scores = stats.zscore(data[col].dropna())
                outlier_indices = data[col][(z_scores > 3) | (z_scores < -3)].index
                outliers[col] = data.loc[outlier_indices, col]
                if not outliers[col].empty:
                    st.write(f"Outliers in {col}:")
                    st.write(outliers[col])
                else:
                    st.write(f"No significant outliers detected in {col}.")

    with st.expander("Time Series Analysis"):
        # Time Series Analysis
        if st.checkbox("Show Time Series Plot", help="Displays line plots for time series data."):
            st.write("### Time Series Plot")
            date_cols = data.select_dtypes(include=['datetime']).columns
            if not date_cols.empty:
                date_col = st.selectbox("Select Date Column", options=date_cols, key='ts_date_col')
                value_cols = data.select_dtypes(include=['float', 'int']).columns
                value_col = st.selectbox("Select Value Column", options=value_cols, key='ts_value_col')
                if date_col and value_col:
                    st.write(f"#### Time Series Plot of {value_col} over {date_col}")
                    fig, ax = plt.subplots()
                    try:
                        sns.lineplot(x=data[date_col], y=data[value_col], ax=ax)
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Could not generate time series plot: {e}")
            else:
                st.write("No datetime columns available for time series analysis.")

    with st.expander("Missing Data Visualization"):
        # Missing Data Heatmap
        if st.checkbox("Show Missing Data Heatmap", help="Visualizes missing data in the dataset."):
            st.write("### Missing Data Heatmap")
            try:
                import missingno as msno
                fig, ax = plt.subplots()
                msno.heatmap(data, ax=ax)
                st.pyplot(fig)
            except ImportError:
                st.error("The 'missingno' library is required for this feature. Install it using 'pip install missingno'.")
            except Exception as e:
                st.error(f"Could not generate missing data heatmap: {e}")

    # LLM Analysis and Suggestions
    st.header("LLM Analysis and Suggestions")

    if st.button("Analyze Data"):
        analysis_response = analyze_data_with_llm(json_data, prompt_type="analysis")
        st.write("### Data Analysis and Improvement Suggestions:")
        st.write(analysis_response)

    if st.button("Suggest Models"):
        model_suggestions_response = analyze_data_with_llm(json_data, prompt_type="model_suggestions")
        st.write("### Model Suggestions:")
        st.write(model_suggestions_response)

    if st.button("Suggest Visualizations"):
        visualization_suggestions_response = analyze_data_with_llm(json_data, prompt_type="visualization_suggestions")
        st.write("### Visualization Suggestions:")
        st.write(visualization_suggestions_response)

    if st.button("Get Feature Engineering Suggestions"):
        feature_eng_response = analyze_data_with_llm(json_data, prompt_type="feature_engineering")
        st.write("### Feature Engineering Suggestions from LLM:")
        st.write(feature_eng_response)

    st.write("### Ask a Question About Your Data")
    user_question = st.text_input("Enter your question:")
    if st.button("Get Answer from LLM"):
        if user_question:
            custom_response = analyze_data_with_llm(json_data, prompt_type="custom", user_question=user_question)
            st.write("### LLM Response:")
            st.write(custom_response)
        else:
            st.write("Please enter a question.")

else:
    # Display a message to prompt the user to upload a file
    st.write("Please upload a CSV file to proceed.")
