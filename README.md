# Data Analysis and EDA with LLM Integration

This Streamlit app allows users to upload CSV datasets, perform Exploratory Data Analysis (EDA), and leverage OpenAI's GPT-4 via LangChain for advanced insights.

## Features

## Data Upload

- CSV Upload: Upload datasets in CSV format.
- Exploratory Data Analysis (EDA)
- Data Overview: View data shape, types, and missing values.
- Descriptive Statistics: Display statistical measures for numerical columns.

## Data Visualization:

- Histograms, Box Plots, Violin Plots, Scatter Plots: Understand data distributions and relationships.
- Correlation Matrix and Heatmap: Analyze correlations between variables.
- Pair Plots: Visualize pairwise relationships.
- Outlier Detection: Identify outliers using Z-score.
- Time Series Analysis: Visualize trends over time.
- Missing Data Visualization: Detect patterns in missing data.

## LLM Integration

- Data Analysis: Use GPT-4 to identify data issues and get improvement suggestions.
- Model Suggestions: Receive recommendations for suitable machine learning models.
- Visualization Suggestions: Get ideas for effective visualizations.
- Feature Engineering: Obtain suggestions to enhance model performance.
- Custom Queries: Ask questions about your data.

## How It Works

- Data Upload: Upload a CSV file.
- Data Processing: Read data and convert date columns if necessary.
- EDA: Explore data using built-in tools.
- LLM Interaction: Request analysis, model suggestions, visualizations, and feature engineering ideas from GPT-4.
- Large Dataset Handling: Option to limit data sent to the LLM to stay within token limits.

## Steps to run code 
```
pip install -r requirements.txt
```
```
OPENAI_API_KEY=your-openai-api-key
```
```
streamlit run app.py
```
