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
    llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)
    
    # Define the prompt for generating multiple visualization codes
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
    code = chain.run(data_sample=data_sample, num_visualizations=num_visualizations)
    
    # Clean markdown code block markers if present
    code = code.strip()
    if code.startswith("```python"):
        code = code[9:]  # Remove the opening ```python
    if code.endswith("```"):
        code = code[:-3]  # Remove the closing ```
    
    return code

# Function to execute the generated code safely with enhanced filtering
def execute_visualization_code(code, data, num_visualizations=3):
    disallowed_patterns = [
        r'\bimport\b', r'\b__\b', r'\bopen\(', r'\beval\(', r'\bexec\(', r'\binput\(', 
        r'\bcompile\(', r'\bos\b', r'\bsys\b', r'\bsubprocess\b', r'\bthreading\b', 
        r'\bgetattr\(', r'\bsetattr\(', r'\bdelattr\(', r'\bglobals\(', r'\blocals\(', 
        r'\bvars\(', r'\bhelp\(',
    ]

    # Check for disallowed patterns
    for pattern in disallowed_patterns:
        if re.search(pattern, code):
            raise ValueError(f"Disallowed pattern '{pattern}' found in code.")

    # Clean code from potential formatting issues
    code = code.strip().replace('\r', '')  # Remove any carriage return characters
    
    # Check if the code is valid syntax
    try:
        ast.parse(code)
    except SyntaxError as e:
        raise SyntaxError(f"Syntax error in generated code on line {e.lineno}: {e.msg} Line content: {e.text}")

    # Safe environment for execution
    safe_globals = {
        'pd': pd,
        'px': px,
        'data': data,
    }
    safe_locals = {}

    # Execute the code
    try:
        exec(code, safe_globals, safe_locals)
    except Exception as e:
        raise RuntimeError(f"Error executing generated code: {e}")

    # Retrieve the figure objects if generated
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
    
    # Only allow selection of categorical columns for filtering
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
    selected_columns = st.sidebar.multiselect("Select Categorical Columns for Filtering", options=categorical_columns)

    # Filter based on user-selected values within each selected categorical column
    for col in selected_columns:
        unique_values = data[col].unique()
        selected_values = st.sidebar.multiselect(f"Select values for '{col}'", options=unique_values, default=unique_values)
        data = data[data[col].isin(selected_values)]

    return data

# Function to convert Plotly figures to base64
def convert_figs_to_base64(figs):
    images_base64 = []
    for fig in figs:
        img_bytes = fig.to_image(format="png", engine="kaleido")
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        images_base64.append(img_base64)
    return images_base64

# Function to send images to LLM for analysis with context
def analyze_visualizations_with_llm(images_base64):
    llm = ChatOpenAI(model_name="gpt-4o-mini")
    
    # Prepare message content with images
    message_content = [
        {
            "type": "text",
            "text": "Analyze the following images for patterns or insights in the data and understand relationships between data points.",
        }
    ]
    
    for img_base64 in images_base64:
        message_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_base64}",
            },
        })
    
    # Create the multimodal message
    message = HumanMessage(content=message_content)
    
    # Send to the model
    response = llm.invoke([message])
    return response.content

# Visualization section in Streamlit
def visualization_section():
    st.header("AI-Powered Interactive Visualizations")

    if 'data' not in st.session_state:
        st.error("Please upload a dataset first.")
        return

    data = st.session_state['data']

    # Option to filter data or use the entire sample
    data_filter_option = st.selectbox("Data Sample Option", ["Use Entire Dataset", "Filter by Categorical Variables"])
    
    if data_filter_option == "Filter by Categorical Variables":
        filtered_data = filter_data(data)
    else:
        filtered_data = data

    # Provide only the top 5 rows of the filtered data to LLM as a sample
    data_sample = filtered_data.head(5).to_csv(index=False)

    st.subheader("AI-Generated Visualizations")

    # Number of visualizations to generate
    num_visualizations = st.slider("Select the number of visualizations", min_value=1, max_value=5, value=3)

    if st.button("Generate Visualizations with AI"):
        with st.spinner("Generating visualizations..."):
            try:
                # Get code from LLM
                code = get_llm_visualization_code(data_sample, num_visualizations)
                st.code(code, language='python')

                # Execute the code on the full filtered dataset
                figs = execute_visualization_code(code, filtered_data, num_visualizations)

                # Display the generated figures
                for idx, fig in enumerate(figs, start=1):
                    st.plotly_chart(fig, use_container_width=True)

                # Convert figures to base64 for LLM analysis
                images_base64 = convert_figs_to_base64(figs)
                
                # Send images to LLM for analysis
                llm_response = analyze_visualizations_with_llm(images_base64)
                
                st.write("LLM Analysis of Visualizations:")
                st.write(llm_response)

            except Exception as e:
                st.error(f"An error occurred: {e}")