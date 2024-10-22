import streamlit as st
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI 
from dotenv import load_dotenv
import os
import base64

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

# Function to sample data for LLM analysis
def sample_data_for_llm(data, max_rows=1000):
    if len(data) > max_rows:
        return data.sample(n=max_rows)
    else:
        return data

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
    json_data = convert_data_to_json(sample_data_for_llm(data))

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

    with st.expander("Data Cleaning Overview"):
        # Columns with High Missing Values
        if st.checkbox("Show Columns with High Missing Values", help="Identifies columns with a high percentage of missing values."):
            st.write("### Columns with High Missing Values")
            missing_percent = data.isnull().mean() * 100
            high_missing = missing_percent[missing_percent > 20]  # Threshold can be adjusted
            st.write(high_missing.sort_values(ascending=False))

        # Columns with Low Variance
        if st.checkbox("Show Columns with Low Variance", help="Identifies columns with low variance that may be dropped."):
            st.write("### Columns with Low Variance")
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold=0.01)
            selector.fit(data.select_dtypes(include=['float', 'int']).fillna(0))
            low_variance_cols = data.select_dtypes(include=['float', 'int']).columns[~selector.get_support()]
            st.write(low_variance_cols)

    with st.expander("Statistical Moments"):
        # Skewness and Kurtosis
        if st.checkbox("Show Skewness and Kurtosis", help="Displays skewness and kurtosis for numerical variables."):
            st.write("### Skewness and Kurtosis")
            numeric_cols = data.select_dtypes(include=['float', 'int']).columns
            skewness = data[numeric_cols].skew()
            kurtosis = data[numeric_cols].kurtosis()
            stats_df = pd.DataFrame({'Skewness': skewness, 'Kurtosis': kurtosis})
            st.write(stats_df)

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
                hue_col = st.selectbox("Select Hue variable (optional)", options=[None] + list(data.columns), key='scatter_hue')
                # Option to sample data
                sample_size = st.number_input("Select number of data points to plot (sampling if necessary)", min_value=100, max_value=len(data), value=1000, step=100)
                plot_data = data.sample(n=sample_size) if len(data) > sample_size else data
                if x_axis and y_axis:
                    st.write(f"#### Scatter Plot between {x_axis} and {y_axis}")
                    fig, ax = plt.subplots()
                    try:
                        sns.scatterplot(
                            x=plot_data[x_axis],
                            y=plot_data[y_axis],
                            hue=plot_data[hue_col] if hue_col else None,
                            ax=ax,
                            alpha=0.6
                        )
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Could not generate scatter plot: {e}")
            else:
                st.write("Not enough numerical columns to create a scatter plot.")

        # Interactive Scatter Plot
        if st.checkbox("Show Interactive Scatter Plot", help="Displays an interactive scatter plot using Plotly."):
            st.write("### Interactive Scatter Plot")
            import plotly.express as px
            numeric_cols = data.select_dtypes(include=['float', 'int']).columns
            if len(numeric_cols) >= 2:
                x_axis = st.selectbox("Select X-axis variable", options=numeric_cols, key='interactive_scatter_x')
                y_axis = st.selectbox("Select Y-axis variable", options=numeric_cols, key='interactive_scatter_y')
                hue_col = st.selectbox("Select Color variable (optional)", options=[None] + list(data.columns), key='interactive_scatter_hue')
                if x_axis and y_axis:
                    fig = px.scatter(data_frame=data, x=x_axis, y=y_axis, color=hue_col)
                    st.plotly_chart(fig)
            else:
                st.write("Not enough numerical columns to create a scatter plot.")

        # Categorical Variable Analysis
        if st.checkbox("Show Value Counts and Bar Plots for Categorical Variables", help="Displays frequency counts and bar plots for categorical variables."):
            st.write("### Categorical Variables Analysis")
            cat_cols = data.select_dtypes(include=['object', 'category']).columns
            for col in cat_cols:
                st.write(f"#### {col}")
                st.write(data[col].value_counts())
                fig, ax = plt.subplots()
                try:
                    sns.countplot(data[col], ax=ax, order=data[col].value_counts().index)
                    plt.xticks(rotation=45, ha='right')
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Could not plot bar plot for {col}: {e}")

    with st.expander("Correlation and Relationships"):
        # Correlation Matrix and Heatmap
        if st.checkbox("Show Correlation Heatmap", help="Displays a heatmap of the correlation matrix for numerical variables."):
            st.write("### Correlation Heatmap")
            numeric_cols = data.select_dtypes(include=['float', 'int']).columns
            if len(numeric_cols) >= 2:
                corr = data[numeric_cols].corr()
                fig, ax = plt.subplots(figsize=(12, 10))  # Increased figure size
                try:
                    sns.heatmap(
                        corr, 
                        annot=True, 
                        fmt=".2f", 
                        cmap='coolwarm', 
                        cbar=True,
                        annot_kws={"size": 8},  # Adjust annotation font size
                        ax=ax
                    )
                    ax.set_xticklabels(
                        ax.get_xticklabels(),
                        rotation=45,
                        horizontalalignment='right',
                        fontsize=8  # Adjust x-axis tick label size
                    )
                    ax.set_yticklabels(
                        ax.get_yticklabels(),
                        rotation=0,
                        fontsize=8  # Adjust y-axis tick label size
                    )
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

    with st.expander("Normality Checks"):
        # Normality Checks using QQ Plots
        if st.checkbox("Show QQ Plots", help="Displays QQ plots for numerical variables to assess normality."):
            st.write("### QQ Plots")
            import statsmodels.api as sm
            numeric_cols = data.select_dtypes(include=['float', 'int']).columns
            for col in numeric_cols:
                st.write(f"#### QQ Plot for {col}")
                fig = sm.qqplot(data[col].dropna(), line='s')
                st.pyplot(fig)

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

    with st.expander("Data Filtering"):
        st.write("### Filter Data")
        filter_col = st.selectbox("Select column to filter", options=data.columns)
        if filter_col:
            if data[filter_col].dtype == 'object' or data[filter_col].dtype.name == 'category':
                filter_vals = st.multiselect(f"Select values for {filter_col}", options=data[filter_col].unique())
                if filter_vals:
                    data = data[data[filter_col].isin(filter_vals)]
            else:
                min_val = float(data[filter_col].min())
                max_val = float(data[filter_col].max())
                range_vals = st.slider(f"Select range for {filter_col}", min_value=min_val, max_value=max_val, value=(min_val, max_val))
                data = data[(data[filter_col] >= range_vals[0]) & (data[filter_col] <= range_vals[1])]
        st.write("#### Filtered Data")
        st.dataframe(data)

    # Data Preprocessing Options
    # st.header("Data Preprocessing")
    # with st.expander("Data Preprocessing Options"):
    #     if st.checkbox("Handle Missing Values"):
    #         st.write("### Missing Values Imputation")
    #         impute_method = st.selectbox("Select imputation method", options=['Mean', 'Median', 'Mode'])
    #         numeric_cols = data.select_dtypes(include=['float', 'int']).columns
    #         for col in numeric_cols:
    #             if data[col].isnull().sum() > 0:
    #                 if impute_method == 'Mean':
    #                     data[col].fillna(data[col].mean(), inplace=True)
    #                 elif impute_method == 'Median':
    #                     data[col].fillna(data[col].median(), inplace=True)
    #                 elif impute_method == 'Mode':
    #                     data[col].fillna(data[col].mode()[0], inplace=True)
    #         st.write("Missing values have been imputed.")

    #     if st.checkbox("Encode Categorical Variables"):
    #         st.write("### Encoding Categorical Variables")
    #         encoding_method = st.selectbox("Select encoding method", options=['One-Hot Encoding', 'Label Encoding'])
    #         cat_cols = data.select_dtypes(include=['object', 'category']).columns
    #         if encoding_method == 'One-Hot Encoding':
    #             data = pd.get_dummies(data, columns=cat_cols)
    #         elif encoding_method == 'Label Encoding':
    #             from sklearn.preprocessing import LabelEncoder
    #             label_encoders = {}
    #             for col in cat_cols:
    #                 le = LabelEncoder()
    #                 data[col] = le.fit_transform(data[col].astype(str))
    #                 label_encoders[col] = le
    #         st.write("Categorical variables have been encoded.")

    #     if st.checkbox("Scale Numerical Features"):
    #         st.write("### Feature Scaling")
    #         scaling_method = st.selectbox("Select scaling method", options=['Standardization', 'Normalization'])
    #         numeric_cols = data.select_dtypes(include=['float', 'int']).columns
    #         if scaling_method == 'Standardization':
    #             from sklearn.preprocessing import StandardScaler
    #             scaler = StandardScaler()
    #             data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    #         elif scaling_method == 'Normalization':
    #             from sklearn.preprocessing import MinMaxScaler
    #             scaler = MinMaxScaler()
    #             data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    #         st.write("Numerical features have been scaled.")

    # EDA Report Generation
    # st.header("EDA Report Generation")
    # if st.button("Generate EDA Report"):
    #     st.write("### EDA Report")
    #     try:
    #         from pandas_profiling import ProfileReport
    #         from streamlit_pandas_profiling import st_profile_report
    #         profile = ProfileReport(data, minimal=True)
    #         st_profile_report(profile)
    #     except Exception as e:
    #         st.error(f"Could not generate EDA report: {e}")
    #         st.info("You may need to install pandas-profiling and streamlit_pandas_profiling.")

    # Data Export
    # st.write("### Download Processed Data")
    # csv = data.to_csv(index=False)
    # b64 = base64.b64encode(csv.encode()).decode()  # B64 encode
    # href = f'<a href="data:file/csv;base64,{b64}" download="processed_data.csv">Download CSV File</a>'
    # st.markdown(href, unsafe_allow_html=True)

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
