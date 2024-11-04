import streamlit as st
import pandas as pd
import plotly.express as px
import ast
import re
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

# Function to get LLM-generated code for multiple visualizations
def get_llm_visualization_code(data_sample, num_visualizations=3):
    # Initialize the LLM with GPT-4 model
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
- The code should not contain import statements or data loading code.
- Do not include any code for filesystem access, OS or system-level functions, or network operations.
- Avoid any advanced or non-standard Python constructs; stick strictly to simple and safe Plotly figure generation commands.

Provide only the safe code within these constraints. Do not include any explanations or comments.
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    code = chain.run(data_sample=data_sample, num_visualizations=num_visualizations)
    return code.strip()

# Function to execute the generated code safely with enhanced filtering
def execute_visualization_code(code, data, num_visualizations=3):
    # Disallowed patterns to search for
    disallowed_patterns = [
        r'\bimport\b',         # Disallow any import statements
        r'\b__\b',             # Disallow usage of double underscores
        r'\bopen\(',           # Disallow open function
        r'\beval\(',           # Disallow eval function
        r'\bexec\(',           # Disallow exec function
        r'\binput\(',          # Disallow input function
        r'\bcompile\(',        # Disallow compile function
        r'\bos\b',             # Disallow os module
        r'\bsys\b',            # Disallow sys module
        r'\bsubprocess\b',     # Disallow subprocess module
        r'\bthreading\b',      # Disallow threading module
        r'\bgetattr\(',        # Disallow getattr function
        r'\bsetattr\(',        # Disallow setattr function
        r'\bdelattr\(',        # Disallow delattr function
        r'\bglobals\(',        # Disallow globals function
        r'\blocals\(',         # Disallow locals function
        r'\bvars\(',           # Disallow vars function
        r'\bhelp\(',           # Disallow help function
    ]

    for pattern in disallowed_patterns:
        if re.search(pattern, code):
            raise ValueError(f"Disallowed pattern '{pattern}' found in code.")

    # Parse the code to check for syntax errors
    try:
        ast.parse(code)
    except SyntaxError as e:
        raise SyntaxError(f"Syntax error in generated code: {e}")

    # Define a restricted execution environment
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

# Visualization section in Streamlit
def visualization_section():
    st.header("AI-Powered Interactive Visualizations")

    if 'data' not in st.session_state:
        st.error("Please upload a dataset first.")
        return

    data = st.session_state['data']
    data_sample = data.head(5).to_csv(index=False)

    st.subheader("AI-Generated Visualizations")

    num_visualizations = st.slider("Select the number of visualizations", min_value=1, max_value=5, value=3)

    if st.button("Generate Visualizations with AI"):
        with st.spinner("Generating visualizations..."):
            try:
                # Get code from LLM
                code = get_llm_visualization_code(data_sample, num_visualizations)
                st.code(code, language='python')

                # Execute the code safely
                figs = execute_visualization_code(code, data, num_visualizations)

                # Display the generated figures
                for idx, fig in enumerate(figs, start=1):
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"An error occurred: {e}")

