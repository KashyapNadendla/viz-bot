# Data Analysis and EDA with LLM Integration

This Streamlit app allows users to upload CSV datasets, perform Exploratory Data Analysis (EDA), and leverage OpenAI's GPT-4 via LangChain for advanced insights with **enhanced chat-based visualization capabilities**.

## 🚀 New Features

### 🤖 Multi-Agent AI Visualization System
- **Four Specialized Agents**: Code Generator, Code Reviewer, Visualization Interpreter, and Discussion Facilitator
- **Agentic Architecture**: Each agent has a specific role and collaborates with others
- **Natural Language Chart Creation**: Request charts using plain English (e.g., "Show me a scatter plot of sales vs profit")
- **Conversational Memory**: The system remembers your previous requests and builds context
- **Persistent Visualizations**: Charts and interpretations are stored and persist across tab changes
- **Smart Suggestions**: Get AI-powered visualization ideas based on your data and previous charts
- **Interactive Chat**: Ask follow-up questions about your charts and data
- **Context-Aware Responses**: Each interaction makes the system smarter and more personalized
- **Visualization History**: Access all previously created charts in the gallery

### 📊 Enhanced Visualization System
- **Multi-Agent Architecture**: Four specialized AI agents work together
  - Agent 1 (Code Generator): Creates visualization code based on data analysis
  - Agent 2 (Code Reviewer): Reviews code for syntax errors and security issues
  - Agent 3 (Visualization Interpreter): Analyzes generated charts and extracts insights
  - Agent 4 (Discussion Facilitator): Coordinates all agents and provides final insights
- **Memory-Enhanced Chat**: Maintains conversation history and context
- **Persistent Storage**: Visualizations and interpretations are stored and accessible across sessions
- **Quick Charts**: Pre-built templates for common visualizations
- **Real-time Insights**: Get instant analysis of generated charts
- **Visualization Gallery**: View all previously created charts with their insights



## Features

### Data Upload
- CSV Upload: Upload datasets in CSV format
- Exploratory Data Analysis (EDA)
- Data Overview: View data shape, types, and missing values
- Descriptive Statistics: Display statistical measures for numerical columns

### Data Visualization
- **AI-Generated Visualizations**: Let AI create custom charts based on your data
- **Quick Charts**: Pre-built templates (Bar, Line, Scatter, Histogram, Box, Violin, Heatmap, Pie)
- **Interactive Chat Assistant**: Ask questions and get insights about your data
- **Histograms, Box Plots, Violin Plots, Scatter Plots**: Understand data distributions and relationships
- **Correlation Matrix and Heatmap**: Analyze correlations between variables
- **Pair Plots**: Visualize pairwise relationships
- **Outlier Detection**: Identify outliers using Z-score
- **Time Series Analysis**: Visualize trends over time
- **Missing Data Visualization**: Detect patterns in missing data



### LLM Integration
- **Conversational AI**: Chat with AI about your data and visualizations
- **Memory System**: The AI remembers your previous interactions and builds context
- **Data Analysis**: Use GPT-4 to identify data issues and get improvement suggestions
- **Model Suggestions**: Receive recommendations for suitable machine learning models
- **Visualization Suggestions**: Get ideas for effective visualizations
- **Feature Engineering**: Obtain suggestions to enhance model performance
- **Custom Queries**: Ask questions about your data

## How It Works

### For Structured Data (CSV):
1. **Data Upload**: Upload a CSV file
2. **Data Processing**: Read data and convert date columns if necessary
3. **EDA**: Explore data using built-in tools
4. **AI Chart Assistant**: 
   - Request charts using natural language
   - Get AI-generated visualizations with insights
   - Chat with the system to explore your data
   - Build context through conversation
5. **LLM Interaction**: Request analysis, model suggestions, visualizations, and feature engineering ideas from GPT-4
6. **Memory System**: The application maintains conversation history and context for personalized interactions



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
