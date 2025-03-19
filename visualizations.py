# visualizations.py

import streamlit as st
import pandas as pd
import plotly.express as px
import ast
import re
import base64
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage

##############################################
# Agent 1: Code Generator (Improved with Dataset Stats)
##############################################
def generate_visualization_code(data_sample, dataset_stats, num_visualizations=3):
    """
    Uses gpt-4o-mini to generate secure Python code (without import statements)
    that creates a specified number of Plotly Express figures (fig1, fig2, ...).
    The dataset statistics are included in the prompt to generate more meaningful visualizations.
    """
    llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0.5)
    prompt_template = PromptTemplate(
        input_variables=["data_sample", "dataset_stats", "num_visualizations"],
        template="""
You are a data visualization expert with strong statistical knowledge. Given the following dataset sample in CSV format and its statistics:

### Dataset Sample:
{data_sample}

### Dataset Statistics:
{dataset_stats}

Generate Python code using Plotly Express to create {num_visualizations} different and insightful visualizations for this data.
Assume that the data is already loaded into a pandas DataFrame named 'data'.

Follow these strict guidelines:
- Define exactly {num_visualizations} Plotly figure objects named 'fig1', 'fig2', ..., 'fig{num_visualizations}'.
- Do not include any import statements or data-loading code.
- Do not include filesystem, OS, or network operations.
- NO IMPORT STATEMENTS IN THE CODE.
- Use only simple and safe Plotly Express commands.
- Provide only the code with no explanations or comments.

Code:
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    code = chain.run(data_sample=data_sample, dataset_stats=dataset_stats, num_visualizations=num_visualizations)
    code = code.strip()
    if code.startswith("```python"):
        code = code[len("```python"):].strip()
    if code.endswith("```"):
        code = code[:-3].strip()
    return code

##############################################
# Agent 2: Code Reviewer (Syntax Checker)
##############################################
def review_code_syntax(code):
    """
    Uses gpt-4o-mini to review the generated code strictly for syntax errors.
    If the code is correct, the agent should respond with:
      "No syntax errors detected."
    Otherwise, it should list the syntax issues.
    """
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    prompt_template = PromptTemplate(
        input_variables=["code"],
        template="""
You are a Python code reviewer specialized in syntax analysis. Please review the following Python code strictly for syntax errors.
If the code is syntactically correct, respond with exactly "No syntax errors detected." 
If there are syntax errors, list them briefly.

Code:
{code}
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    review = chain.run(code=code)
    return review.strip()

##############################################
# Code Execution (with Safety Checks)
##############################################
def execute_visualization_code(code, data, num_visualizations=3):
    """
    Executes the provided code after performing extra safety checks.
    Raises an error if disallowed patterns are detected or if syntax issues exist.
    """
    disallowed_patterns = [
        r'\bimport\b', r'\b__\b', r'\bopen\(', r'\beval\(', r'\bexec\(', r'\binput\(',
        r'\bcompile\(', r'\bos\b', r'\bsys\b', r'\bsubprocess\b', r'\bthreading\b',
        r'\bgetattr\(', r'\bsetattr\(', r'\bdelattr\(', r'\bglobals\(', r'\blocals\(',
        r'\bvars\(', r'\bhelp\(',
    ]
    for pattern in disallowed_patterns:
        if re.search(pattern, code):
            raise ValueError(f"Disallowed pattern '{pattern}' found in code.")
    code_clean = code.strip().replace('\r', '')
    try:
        ast.parse(code_clean)
    except SyntaxError as e:
        raise SyntaxError(f"Syntax error detected during parsing on line {e.lineno}: {e.msg}")
    safe_globals = {
        'pd': pd,
        'px': px,
        'data': data,
    }
    safe_locals = {}
    try:
        exec(code_clean, safe_globals, safe_locals)
    except Exception as e:
        raise RuntimeError(f"Error executing generated code: {e}")
    figs = []
    for i in range(1, num_visualizations + 1):
        fig = safe_locals.get(f'fig{i}')
        if fig is None:
            raise ValueError(f"The generated code did not produce a 'fig{i}' object.")
        figs.append(fig)
    return figs

##############################################
# Agent 3: Visualization Interpreter (Enhanced)
##############################################
def interpret_visualizations(figs, data_summary):
    """
    Uses a multimodal approach to analyze visualizations by sending both
    the dataset summary and base64-encoded images to GPT-4o-mini.
    """
    images_base64 = []
    for fig in figs:
        img_bytes = fig.to_image(format="png", engine="kaleido")
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        images_base64.append(img_base64)
        
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)
    
    # Prepare the multimodal input for LLM
    message_content = [
        {
            "type": "text",
            "text": f"""
You are an expert data scientist specializing in **statistical pattern detection, anomaly recognition, and visualization optimization**.
You will receive:
1. **A dataset summary**: Contains key numerical statistics and column names.
2. **Base64-encoded images of visualizations**.

**Your task**:
- **Analyze the images** and detect **trends, distributions, correlations, and anomalies** in the data.
- **Cross-reference the dataset summary** to validate key **statistical findings**.
- **Explain why certain patterns emerge** and how they impact business or research decisions.
- **Suggest improvements**: If the visualization does not effectively convey the insights, recommend better alternatives.

---
### **Dataset Summary:**
{data_summary}

Now analyze the following visualizations and extract meaningful insights:
"""
        }
    ]
    
    # Attach the images
    for img_base64 in images_base64:
        message_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_base64}",
            },
        })

    # Send multimodal message
    message = HumanMessage(content=message_content)
    response = llm.invoke([message])
    
    return response.content.strip()

##############################################
# Enhanced Multi-Agent Discussion (Main Page)
##############################################
def agents_discussion(generated_code, review_feedback, interpretation):
    """
    Simulate a collaborative discussion among four specialized agents:
    - Code Generator: provided the visualization code.
    - Code Reviewer: provided the syntax review.
    - Visualization Interpreter: provided the interpretation based on the visuals.
    - Data Analysis Agent: an expert data scientist who analyzes the underlying data patterns.
    
    The agents discuss discrepancies, potential improvements, and recommendations.
    """
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)
    prompt_template = PromptTemplate(
        input_variables=["generated_code", "review_feedback", "interpretation"],
        template="""
You are facilitating a discussion among four specialized agents:
- Code Generator: provided the following visualization code.
- Code Reviewer: provided the following syntax review.
- Visualization Interpreter: provided the following interpretation of the generated visualizations.
- Data Analysis Agent: an expert data scientist who analyzes underlying data patterns, trends, and correlations.

Code Generator Output:
{generated_code}

Code Reviewer Feedback:
{review_feedback}

Visualization Interpreter Analysis:
{interpretation}

Discuss any discrepancies, potential improvements, and clarifications. Then provide a final consolidated summary that combines these insights and recommendations for enhancing the overall data analysis process.
Final Summary:
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    discussion = chain.run(generated_code=generated_code, review_feedback=review_feedback, interpretation=interpretation)
    return discussion.strip()

##############################################
# Chat Discussion for Sidebar (Visualization Interpreter + Data Analysis Agent)
##############################################
def chat_discussion(user_query, context_info, chat_history):
    """
    Simulate a discussion between two agents for answering follow-up questions:
    - Agent 1 (Visualization Interpreter): Provides insights based on the visualizations.
    - Agent 2 (Data Analysis Agent): Provides an in-depth analysis of data trends, anomalies, and patterns.
    
    Uses context information and chat history to produce a final consolidated answer.
    """
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)
    history_str = "\n".join(chat_history) if chat_history else "No previous conversation."
    prompt_template = PromptTemplate(
        input_variables=["user_query", "context_info", "history"],
        template="""
You are facilitating a discussion between two agents:
Agent 1 (Visualization Interpreter): Provides insights based on the visualizations.
Agent 2 (Data Analysis Agent): Provides an in-depth analysis of data trends, anomalies, and patterns.

Context Information:
{context_info}

Conversation History:
{history}

User Query:
{user_query}

Simulate a discussion between Agent 1 and Agent 2, and provide a final consolidated answer to the user's query.
Final Answer:
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    answer = chain.run(user_query=user_query, context_info=context_info, history=history_str)
    return answer.strip()

##############################################
# Conversational Chatbot Interface (Sidebar)
##############################################
def chatbot_interface():
    st.sidebar.header("Interactive Chatbot")
    
    # Initialize chat history in session state if needed
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history (only user queries and AI responses)
    with st.sidebar.container():
        st.markdown("### Conversation History")
        for message in st.session_state.chat_history:
            st.markdown(message)
    
    # User input for the chatbot
    user_input = st.sidebar.text_input("Your question:", key="user_input_chat")
    if st.sidebar.button("Send", key="chat_send"):
        if user_input:
            context_info = st.session_state.get("context_info", "No context provided.")
            answer = chat_discussion(user_input, context_info, st.session_state.chat_history)
            # Append only the user query and AI answer to the chat history
            st.session_state.chat_history.append(f"**User:** {user_input}")
            st.session_state.chat_history.append(f"**AI:** {answer}")
            st.rerun()
        else:
            st.sidebar.warning("Please enter a question.")

##############################################
# Utility: Data Filtering
##############################################
def filter_data(data):
    st.sidebar.header("Data Filtering Options")
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
    selected_columns = st.sidebar.multiselect("Select Categorical Columns for Filtering", options=categorical_columns)
    for col in selected_columns:
        unique_values = data[col].unique()
        selected_values = st.sidebar.multiselect(f"Select values for '{col}'", options=unique_values, default=unique_values)
        data = data[data[col].isin(selected_values)]
    return data

##############################################
# Main Visualization Section with Chatbot Interface
##############################################
def visualization_section():
    st.header("AI-Powered Interactive Visualizations")
    
    if 'data' not in st.session_state:
        st.error("Please upload a dataset first.")
        return
    data = st.session_state['data']
    
    # Optionally filter data
    data_filter_option = st.selectbox("Data Sample Option", ["Use Entire Dataset", "Filter by Categorical Variables"])
    if data_filter_option == "Filter by Categorical Variables":
        filtered_data = filter_data(data)
    else:
        filtered_data = data
    
    # Create a dataset summary for interpretation and context
    data_summary = f"Dataset Summary:\nRows: {filtered_data.shape[0]}, Columns: {filtered_data.shape[1]}\nColumns: {', '.join(filtered_data.columns)}"
    
    # Use only the top 5 rows as a sample for code generation
    data_sample = filtered_data.head(5).to_csv(index=False)
    
    st.subheader("Step 1: Generate Visualization Code")
    num_visualizations = st.slider("Select the number of visualizations", min_value=1, max_value=5, value=3)
    
    if st.button("Generate Visualizations with AI"):
        with st.spinner("Generating visualization code..."):
            try:
                # Agent 1: Generate visualization code (with dataset stats)
                generated_code = generate_visualization_code(data_sample, data_summary, num_visualizations)
                st.subheader("Generated Code")
                st.code(generated_code, language='python')
                
                # Agent 2: Code review (syntax check)
                st.subheader("Code Review (Syntax Check)")
                review_feedback = review_code_syntax(generated_code)
                st.write(review_feedback)
                if "no syntax errors detected" not in review_feedback.lower():
                    st.error("Code review detected potential syntax issues. Please try again.")
                    return
                
                # Execute the generated code safely
                st.subheader("Executing Visualization Code")
                figs = execute_visualization_code(generated_code, filtered_data, num_visualizations)
                for idx, fig in enumerate(figs, start=1):
                    st.plotly_chart(fig, use_container_width=True)
                
                # Agent 3: Enhanced interpretation of visualizations (data-centric)
                st.subheader("Visualization Interpretation")
                interpretation = interpret_visualizations(figs, data_summary)
                st.write(interpretation)
                
                # Enhanced Multi-Agent Discussion (displayed on main page)
                st.subheader("Agents Discussion")
                discussion = agents_discussion(generated_code, review_feedback, interpretation)
                st.write(discussion)
                
                # Save context information for follow-up queries (without including the full interpretation)
                context_info = f"Dataset Summary: {filtered_data.shape[0]} rows, {filtered_data.shape[1]} columns\n"
                context_info += "Columns: " + ", ".join(filtered_data.columns)
                context_info += "\nAgents Discussion Summary: " + discussion
                st.session_state.context_info = context_info
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
    
    # Always show the chatbot interface on the sidebar (the main page content remains visible)
    chatbot_interface()
