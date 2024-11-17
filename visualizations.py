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

# Function to get LLM-generated code for multiple visualizations
def get_llm_visualization_code(data_sample, num_visualizations=3):
    llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0.5)
    prompt_template = PromptTemplate(
        input_variables=["data_sample", "num_visualizations"],
        template="""
You are a data visualization expert. Given the following data sample in CSV format:

{data_sample}

Generate Python code using Plotly Express to create {num_visualizations} different and insightful visualizations for this data.
Assume that the data is already loaded into a pandas DataFrame named 'data'.

Generate only the secure code. Follow these restrictions strictly:
- The code should define {num_visualizations} Plotly figure objects named 'fig1', 'fig2', ..., 'fig{num_visualizations}'.
- The code should not contain import statements or data loading code. Assume Plotly Express is already imported as 'px'.
- Do not include any code for filesystem access, OS or system-level functions, or network operations.
- Avoid any advanced or non-standard Python constructs; stick strictly to simple and safe Plotly figure generation commands.

Provide only the safe code within these constraints. Do not include any explanations or comments.
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    code = chain.run(data_sample=data_sample, num_visualizations=num_visualizations).strip()
    if code.startswith("```python"):
        code = code[9:]
    if code.endswith("```"):
        code = code[:-3]
    return code

# Function to execute the generated code safely
def execute_visualization_code(code, data, num_visualizations=3):
    disallowed_patterns = [
        r'\bimport\b', r'\b__\b', r'\bopen\(', r'\beval\(', r'\bexec\(', 
        r'\binput\(', r'\bcompile\(', r'\bos\b', r'\bsys\b', r'\bsubprocess\b', 
        r'\bthreading\b', r'\bgetattr\(', r'\bsetattr\(', r'\bdelattr\(', 
        r'\bglobals\(', r'\blocals\(', r'\bvars\(', r'\bhelp\(',
    ]
    for pattern in disallowed_patterns:
        if re.search(pattern, code):
            raise ValueError(f"Disallowed pattern '{pattern}' found in code.")

    code = code.strip().replace('\r', '')
    try:
        ast.parse(code)
    except SyntaxError as e:
        raise SyntaxError(f"Syntax error in generated code on line {e.lineno}: {e.msg} Line content: {e.text}")

    safe_globals = {'pd': pd, 'px': px, 'data': data}
    safe_locals = {}

    try:
        exec(code, safe_globals, safe_locals)
    except Exception as e:
        raise RuntimeError(f"Error executing generated code: {e}")

    figs = []
    for i in range(1, num_visualizations + 1):
        fig = safe_locals.get(f'fig{i}')
        if fig is None:
            raise ValueError(f"The generated code did not produce a 'fig{i}' object.")
        figs.append(fig)
    return figs

# Function to filter data based on selected categorical columns and values
def filter_data(data):
    st.sidebar.header("Data Filtering Options")
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
    selected_columns = st.sidebar.multiselect("Select Categorical Columns for Filtering", options=categorical_columns)

    for col in selected_columns:
        unique_values = data[col].unique()
        selected_values = st.sidebar.multiselect(f"Select values for '{col}'", options=unique_values, default=unique_values)
        data = data[data[col].isin(selected_values)]

    return data

# Function to convert Plotly figures to base64
def convert_figs_to_base64(figs):
    images_base64 = []
    for fig in figs:
        try:
            img_bytes = fig.to_image(format="png", engine="kaleido")
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            images_base64.append(img_base64)
        except Exception as e:
            st.error(f"Error converting figure to base64: {e}")
    return images_base64

# Function to send images to LLM for analysis with context
def analyze_visualizations_with_llm(images_base64):
    llm = ChatOpenAI(model_name="gpt-4o-mini")
    message_content = [
        {"type": "text", "text": "Analyze the following images for insights and patterns."}
    ]
    for img_base64 in images_base64:
        message_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_base64}"}
        })

    if not images_base64:
        st.error("No images were available for LLM analysis.")
        return "No analysis performed due to missing images."

    try:
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        st.error(f"Error during LLM analysis: {e}")
        return "Error during LLM analysis."

# Visualization section in Streamlit
def visualization_section():
    st.header("AI-Powered Interactive Visualizations")
    if 'data' not in st.session_state:
        st.error("Please upload a dataset first.")
        return

    data = st.session_state['data']
    data_filter_option = st.selectbox("Data Sample Option", ["Use Entire Dataset", "Filter by Categorical Variables"])
    filtered_data = filter_data(data) if data_filter_option == "Filter by Categorical Variables" else data

    data_sample = filtered_data.head(5).to_csv(index=False)
    st.subheader("AI-Generated Visualizations")
    num_visualizations = st.slider("Select the number of visualizations", min_value=1, max_value=5, value=3)

    if st.button("Generate Visualizations with AI"):
        with st.spinner("Generating visualizations..."):
            try:
                code = get_llm_visualization_code(data_sample, num_visualizations)
                st.code(code, language='python')
                figs = execute_visualization_code(code, filtered_data, num_visualizations)

                for idx, fig in enumerate(figs, start=1):
                    st.plotly_chart(fig, use_container_width=True)

                images_base64 = convert_figs_to_base64(figs)
                llm_response = analyze_visualizations_with_llm(images_base64)
                st.subheader("LLM Analysis of Visualizations")
                st.write(llm_response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
