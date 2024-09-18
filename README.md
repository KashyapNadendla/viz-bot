# Data Analysis and Machine Learning Model Suggestion with LangChain and OpenAI

This Streamlit application allows users to upload a CSV dataset, and with the help of OpenAI's GPT-4 model, it provides insights into the data and suggests appropriate machine learning models for analysis. The app leverages LangChain to interact with the LLM, converting the dataset into a JSON format before sending it to the model for analysis. Additionally, the app can suggest visualizations that would best suit the dataset based on its structure.

### Features

- Upload a CSV File: Users can upload datasets in CSV format.
- Data Analysis: The app analyzes the dataset for issues such as missing values, inconsistencies, or outliers.
- Model Suggestions: Based on the dataset, the app recommends suitable machine learning models for further analysis.
- Visualization Suggestions: The app provides suggestions on the best visualizations to explore the data.
- Handles Large Datasets: If the dataset has more than 10,000 rows, the user can choose to process only the first 500 rows for faster analysis.

### Technologies Used: 

- Streamlit: For building the user interface and handling file uploads.

- LangChain: For constructing prompts and interacting with the OpenAI language model.

- OpenAI GPT-4: For performing data analysis and suggesting machine learning models and visualizations.

- Pandas: For loading and manipulating CSV data.

- FAISS/ScaNN (optional): For efficient similarity search in larger applications involving vector embeddings (if extended).

### How It Works:

- Data Upload: Users upload a CSV file containing the dataset to be analyzed.
- Data Processing: The app converts the dataset into a JSON format.
- Analysis & Suggestions:
- The JSON-formatted dataset is sent to the OpenAI GPT-4 model via LangChain to analyze the data for potential issues or inconsistencies.
- The user can also ask the model to suggest the best machine learning models based on the dataset's structure.
- Additionally, the app can suggest appropriate visualizations to help users explore the data.
- Handling Large Datasets: For datasets with more than 10,000 rows, the app provides an option to limit the analysis to the first 500 rows.

