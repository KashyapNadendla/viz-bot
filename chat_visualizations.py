# chat_visualizations.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ast
import re
import base64
import json
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

class ChatVisualizationSystem:
    def __init__(self):
        self.llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0.7)
        self.memory = ConversationBufferMemory(return_messages=True)
        self.conversation_chain = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=False
        )
        
    def get_data_context(self, data):
        """Generate comprehensive data context for the LLM"""
        context = {
            "shape": data.shape,
            "columns": list(data.columns),
            "dtypes": data.dtypes.to_dict(),
            "numeric_columns": list(data.select_dtypes(include=['number']).columns),
            "categorical_columns": list(data.select_dtypes(include=['object', 'category']).columns),
            "datetime_columns": list(data.select_dtypes(include=['datetime']).columns),
            "sample_data": data.head(10).to_dict('records'),
            "missing_values": data.isnull().sum().to_dict(),
            "basic_stats": data.describe().to_dict() if len(data.select_dtypes(include=['number'])) > 0 else {}
        }
        return context
    
    def store_visualization_context(self, user_request, fig, insights, data_context):
        """Store visualization context for future reference"""
        viz_context = {
            "user_request": user_request,
            "chart_type": type(fig).__name__,
            "insights": insights,
            "data_context": data_context,
            "timestamp": datetime.now(),
            "chart_data": {
                "x_axis": fig.data[0].x if hasattr(fig.data[0], 'x') else None,
                "y_axis": fig.data[0].y if hasattr(fig.data[0], 'y') else None,
                "chart_type": fig.data[0].type if fig.data else None
            }
        }
        return viz_context
    
    def generate_chart_from_request(self, user_request, data, context):
        """Generate a chart based on user's natural language request"""
        
        # Create a comprehensive prompt for chart generation
        prompt_template = PromptTemplate(
            input_variables=["user_request", "shape", "columns", "numeric_columns", "categorical_columns", "datetime_columns", "data_sample"],
            template="""
You are an expert data visualization specialist. A user wants to create a chart based on their request.

User Request: {user_request}

Data Context:
- Dataset shape: {shape}
- Columns: {columns}
- Numeric columns: {numeric_columns}
- Categorical columns: {categorical_columns}
- Datetime columns: {datetime_columns}
- Sample data: {data_sample}

Based on the user's request and the available data, generate Python code to create the most appropriate visualization using Plotly Express (px) or Plotly Graph Objects (go).

Requirements:
1. Use only the available columns from the data
2. Create a meaningful and insightful visualization
3. Use appropriate chart types (scatter, bar, line, histogram, box, violin, heatmap, etc.)
4. Include proper titles, labels, and styling
5. Handle any data preprocessing needed (grouping, aggregating, etc.)
6. Return only the Python code that creates the figure object named 'fig'

Code:
"""
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        code = chain.run({
            "user_request": user_request,
            "shape": context["shape"],
            "columns": context["columns"],
            "numeric_columns": context["numeric_columns"],
            "categorical_columns": context["categorical_columns"],
            "datetime_columns": context["datetime_columns"],
            "data_sample": json.dumps(context["sample_data"][:3])  # Limit sample size
        })
        
        # Clean the code
        code = code.strip()
        if code.startswith("```python"):
            code = code[len("```python"):].strip()
        if code.endswith("```"):
            code = code[:-3].strip()
        
        return code
    
    def execute_chart_code(self, code, data):
        """Safely execute the generated chart code"""
        # Safety checks
        disallowed_patterns = [
            r'\bimport\b', r'\b__\b', r'\bopen\(', r'\beval\(', r'\bexec\(', r'\binput\(',
            r'\bcompile\(', r'\bos\b', r'\bsys\b', r'\bsubprocess\b', r'\bthreading\b',
            r'\bgetattr\(', r'\bsetattr\(', r'\bdelattr\(', r'\bglobals\(', r'\blocals\(',
            r'\bvars\(', r'\bhelp\(',
        ]
        
        for pattern in disallowed_patterns:
            if re.search(pattern, code):
                raise ValueError(f"Disallowed pattern '{pattern}' found in code.")
        
        # Parse for syntax errors
        try:
            ast.parse(code)
        except SyntaxError as e:
            raise SyntaxError(f"Syntax error: {e}")
        
        # Execute safely
        safe_globals = {
            'pd': pd,
            'px': px,
            'go': go,
            'data': data,
        }
        safe_locals = {}
        
        try:
            exec(code, safe_globals, safe_locals)
            fig = safe_locals.get('fig')
            if fig is None:
                raise ValueError("The generated code did not produce a 'fig' object.")
            return fig
        except Exception as e:
            raise RuntimeError(f"Error executing chart code: {e}")
    
    def analyze_chart_insights(self, fig, user_request, data_context, stored_visualizations=None):
        """Analyze the generated chart and provide insights"""
        stored_viz_context = ""
        if stored_visualizations:
            stored_viz_context = "\nPreviously created visualizations:\n"
            for viz in stored_visualizations[-3:]:  # Show last 3 visualizations
                stored_viz_context += f"- {viz['user_request']} ({viz['chart_type']})\n"
        
        prompt_template = PromptTemplate(
            input_variables=["user_request", "data_context", "chart_info", "stored_visualizations"],
            template="""
You are a data analyst. A user requested this chart: "{user_request}"

Data Context:
- Dataset: {data_context[shape][0]} rows, {data_context[shape][1]} columns
- Columns: {data_context[columns]}

Chart Information:
{chart_info}

{stored_visualizations}

Please provide:
1. **Insights**: What patterns, trends, or anomalies do you observe?
2. **Interpretation**: What do these findings mean in the context of the data?
3. **Recommendations**: What additional analysis or visualizations might be helpful?
4. **Questions**: What follow-up questions should the user consider?
5. **Context**: How does this relate to previously created visualizations?

Provide a comprehensive analysis that helps the user understand their data better.
"""
        )
        
        # Get chart information
        chart_info = f"Chart type: {type(fig).__name__}\n"
        if hasattr(fig, 'data') and fig.data:
            chart_info += f"Number of traces: {len(fig.data)}\n"
            for i, trace in enumerate(fig.data):
                chart_info += f"Trace {i+1}: {trace.type} chart\n"
        
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        analysis = chain.run({
            "user_request": user_request,
            "data_context": data_context,
            "chart_info": chart_info,
            "stored_visualizations": stored_viz_context
        })
        
        return analysis.strip()
    
    def get_suggestions(self, data_context, chat_history, stored_visualizations=None):
        """Generate suggestions based on data, conversation history, and stored visualizations"""
        stored_viz_info = ""
        if stored_visualizations:
            stored_viz_info = "\nPreviously created visualizations:\n"
            for viz in stored_visualizations[-3:]:  # Show last 3 visualizations
                stored_viz_info += f"- {viz['user_request']} ({viz['chart_type']})\n"
        
        prompt_template = PromptTemplate(
            input_variables=["data_context", "chat_history", "stored_visualizations"],
            template="""
You are a helpful data visualization assistant. Based on the data context, conversation history, and previously created visualizations, suggest 3-5 interesting visualizations the user might want to explore.

Data Context:
- Dataset: {data_context[shape][0]} rows, {data_context[shape][1]} columns
- Numeric columns: {data_context[numeric_columns]}
- Categorical columns: {data_context[categorical_columns]}
- Datetime columns: {data_context[datetime_columns]}

Recent conversation:
{chat_history}

{stored_visualizations}

Suggest visualizations that would be:
1. Relevant to the data structure
2. Interesting based on the conversation context
3. Complementary to previously created visualizations
4. Likely to reveal meaningful insights

Format each suggestion as a natural language request that the user could type.
"""
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        suggestions = chain.run({
            "data_context": data_context,
            "chat_history": chat_history[-5:] if chat_history else "No previous conversation",
            "stored_visualizations": stored_viz_info
        })
        
        return suggestions.strip()

def chat_visualization_section():
    """Main chat visualization interface"""
    st.header("🤖 AI Chart Assistant")
    st.write("Chat with me to create custom visualizations! I'll remember our conversation and get smarter with each interaction.")
    
    if 'data' not in st.session_state:
        st.error("Please upload a dataset first.")
        return
    
    data = st.session_state['data']
    
    # Initialize the chat system
    if 'chat_viz_system' not in st.session_state:
        st.session_state.chat_viz_system = ChatVisualizationSystem()
    
    # Initialize chat history
    if 'chat_viz_history' not in st.session_state:
        st.session_state.chat_viz_history = []
    
    # Initialize stored visualizations
    if 'stored_visualizations' not in st.session_state:
        st.session_state.stored_visualizations = []
    
    # Initialize data context
    if 'data_context' not in st.session_state:
        st.session_state.data_context = st.session_state.chat_viz_system.get_data_context(data)
    
    # Sidebar for suggestions and stored visualizations
    with st.sidebar:
        st.header("💡 Suggestions")
        if st.button("Get Visualization Ideas"):
            suggestions = st.session_state.chat_viz_system.get_suggestions(
                st.session_state.data_context,
                st.session_state.chat_viz_history,
                st.session_state.stored_visualizations
            )
            st.session_state.suggestions = suggestions
        
        if 'suggestions' in st.session_state:
            st.write("**Try these visualizations:**")
            st.write(st.session_state.suggestions)
        
        # Display stored visualizations
        if st.session_state.stored_visualizations:
            st.header("📊 Created Charts")
            for i, viz in enumerate(st.session_state.stored_visualizations):
                with st.expander(f"Chart {i+1}: {viz['user_request'][:50]}..."):
                    st.write(f"**Request:** {viz['user_request']}")
                    st.write(f"**Type:** {viz['chart_type']}")
                    st.write(f"**Created:** {viz['timestamp'].strftime('%H:%M:%S')}")
                    if 'fig' in viz:
                        st.plotly_chart(viz['fig'], use_container_width=True, key=f"sidebar_chart_{i}")
                    if 'insights' in viz:
                        st.write("**Insights:**")
                        st.write(viz['insights'])
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display chat history
        st.subheader("💬 Conversation")
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.chat_viz_history:
                if message["type"] == "user":
                    st.write(f"**You:** {message['content']}")
                elif message["type"] == "assistant":
                    st.write(f"**AI:** {message['content']}")
                    if "chart" in message:
                        st.plotly_chart(message["chart"], use_container_width=True, key=f"chat_viz_chart_{len(st.session_state.chat_viz_history)}")
                    if "insights" in message:
                        with st.expander("📊 Chart Insights"):
                            st.write(message["insights"])
    
    with col2:
        # Data info panel
        st.subheader("📊 Data Info")
        st.write(f"**Rows:** {data.shape[0]}")
        st.write(f"**Columns:** {data.shape[1]}")
        st.write(f"**Numeric:** {len(st.session_state.data_context['numeric_columns'])}")
        st.write(f"**Categorical:** {len(st.session_state.data_context['categorical_columns'])}")
        
        # Quick data preview
        with st.expander("Data Preview"):
            st.dataframe(data.head())
    
    # User input
    st.subheader("🎯 Request a Chart")
    user_input = st.text_input(
        "Describe the chart you want (e.g., 'Show me a scatter plot of sales vs profit'):",
        placeholder="Create a bar chart showing total sales by region..."
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("🚀 Generate Chart", type="primary"):
            if user_input:
                with st.spinner("Creating your chart..."):
                    try:
                        # Generate chart code
                        code = st.session_state.chat_viz_system.generate_chart_from_request(
                            user_input, 
                            data, 
                            st.session_state.data_context
                        )
                        
                        # Execute the code
                        fig = st.session_state.chat_viz_system.execute_chart_code(code, data)
                        
                        # Analyze insights
                        insights = st.session_state.chat_viz_system.analyze_chart_insights(
                            fig, 
                            user_input, 
                            st.session_state.data_context,
                            st.session_state.stored_visualizations
                        )
                        
                        # Store visualization context
                        viz_context = st.session_state.chat_viz_system.store_visualization_context(
                            user_input, fig, insights, st.session_state.data_context
                        )
                        viz_context['fig'] = fig
                        viz_context['insights'] = insights
                        st.session_state.stored_visualizations.append(viz_context)
                        
                        # Store in conversation history
                        st.session_state.chat_viz_history.append({
                            "type": "user",
                            "content": user_input,
                            "timestamp": datetime.now()
                        })
                        
                        st.session_state.chat_viz_history.append({
                            "type": "assistant",
                            "content": f"I've created a chart based on your request: '{user_input}'",
                            "chart": fig,
                            "insights": insights,
                            "timestamp": datetime.now()
                        })
                        
                        # Update memory with visualization context
                        memory_context = f"Created visualization: {user_input}. Chart type: {type(fig).__name__}. Key insights: {insights[:200]}..."
                        st.session_state.chat_viz_system.memory.save_context(
                            {"input": user_input},
                            {"output": memory_context}
                        )
                        
                        st.success("Chart generated successfully!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error generating chart: {str(e)}")
                        st.session_state.chat_viz_history.append({
                            "type": "assistant",
                            "content": f"Sorry, I couldn't create that chart. Error: {str(e)}",
                            "timestamp": datetime.now()
                        })
            else:
                st.warning("Please enter a chart request.")
    
    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_viz_history = []
            st.session_state.stored_visualizations = []
            st.session_state.chat_viz_system.memory.clear()
            st.success("Chat history and visualizations cleared!")
            st.rerun()
    
    # Display the most recent chart if available
    if st.session_state.chat_viz_history:
        latest_message = st.session_state.chat_viz_history[-1]
        if latest_message["type"] == "assistant" and "chart" in latest_message:
            st.subheader("📈 Latest Chart")
            st.plotly_chart(latest_message["chart"], use_container_width=True, key="latest_chart")
            
            if "insights" in latest_message:
                with st.expander("💡 Chart Analysis"):
                    st.write(latest_message["insights"]) 