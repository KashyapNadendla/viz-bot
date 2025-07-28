# unified_visualizations.py

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

class UnifiedVisualizationSystem:
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
            "data_sample": json.dumps(data.head(3).to_dict('records'))  # Use actual data sample
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
        # Clean the code by removing import statements
        cleaned_code = ""
        for line in code.split('\n'):
            line = line.strip()
            # Skip import statements and empty lines
            if not line.startswith('import ') and not line.startswith('from ') and line:
                cleaned_code += line + '\n'
        
        # Safety checks (excluding import since we already removed them)
        disallowed_patterns = [
            r'\b__\b', r'\bopen\(', r'\beval\(', r'\bexec\(', r'\binput\(',
            r'\bcompile\(', r'\bos\b', r'\bsys\b', r'\bsubprocess\b', r'\bthreading\b',
            r'\bgetattr\(', r'\bsetattr\(', r'\bdelattr\(', r'\bglobals\(', r'\blocals\(',
            r'\bvars\(', r'\bhelp\(',
        ]
        
        for pattern in disallowed_patterns:
            if re.search(pattern, cleaned_code):
                raise ValueError(f"Disallowed pattern '{pattern}' found in code.")
        
        # Parse for syntax errors
        try:
            ast.parse(cleaned_code)
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
            exec(cleaned_code, safe_globals, safe_locals)
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
            input_variables=["user_request", "shape", "columns", "chart_info", "stored_visualizations"],
            template="""
You are a data analyst. A user requested this chart: "{user_request}"

Data Context:
- Dataset: {shape[0]} rows, {shape[1]} columns
- Columns: {columns}

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
            "shape": data_context["shape"],
            "columns": data_context["columns"],
            "chart_info": chart_info,
            "stored_visualizations": stored_viz_context
        })
        
        return analysis.strip()
    
    def agents_discussion(self, generated_code, review_feedback, interpretation):
        """Simulate a collaborative discussion among specialized agents"""
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)
        prompt_template = PromptTemplate(
            input_variables=["generated_code", "review_feedback", "interpretation"],
            template="""
You are Agent 4, the Discussion Facilitator, coordinating a collaborative discussion among four specialized AI agents:

**Agent 1 (Code Generator):** Created the visualization code below
**Agent 2 (Code Reviewer):** Reviewed the code for syntax and security issues
**Agent 3 (Visualization Interpreter):** Analyzed the generated visualizations for insights
**Agent 4 (Discussion Facilitator):** You - coordinating the discussion and providing final insights

**Agent 1's Code Generator Output:**
{generated_code}

**Agent 2's Code Reviewer Feedback:**
{review_feedback}

**Agent 3's Visualization Interpreter Analysis:**
{interpretation}

**Your Task as Agent 4 (Discussion Facilitator):**
1. Facilitate a collaborative discussion among all agents
2. Identify any discrepancies or conflicts between the agents' findings
3. Suggest potential improvements or alternative approaches
4. Provide a final consolidated summary that combines all insights
5. Recommend next steps for further analysis

**Discussion Format:**
- Start with a brief overview of each agent's contribution
- Highlight any disagreements or complementary insights
- Provide a unified final recommendation
- Suggest follow-up analyses or visualizations

**Final Multi-Agent Summary:**
"""
        )
        chain = LLMChain(llm=llm, prompt=prompt_template)
        discussion = chain.run(generated_code=generated_code, review_feedback=review_feedback, interpretation=interpretation)
        return discussion.strip()
    
    def review_code_syntax(self, code):
        """Review the generated code for syntax errors"""
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

def unified_visualization_section():
    """Unified visualization interface combining AI-generated charts with chat and agents discussion"""
    st.header("🤖 Multi-Agent AI Visualization System")
    st.write("Experience the power of collaborative AI agents working together to create, review, and analyze your visualizations!")
    
    # Show the agentic architecture
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info("🤖 **Agent 1: Code Generator**\nCreates visualization code based on your data")
    with col2:
        st.info("🔍 **Agent 2: Code Reviewer**\nReviews code for syntax and security")
    with col3:
        st.info("📊 **Agent 3: Visualization Interpreter**\nAnalyzes charts and extracts insights")
    with col4:
        st.info("💬 **Agent 4: Discussion Facilitator**\nCoordinates all agents and provides final insights")
    
    if 'data' not in st.session_state:
        st.error("Please upload a dataset first.")
        return
    
    data = st.session_state['data']
    
    # Initialize the unified system
    if 'unified_viz_system' not in st.session_state:
        st.session_state.unified_viz_system = UnifiedVisualizationSystem()
    
    # Initialize session state
    if 'unified_chat_history' not in st.session_state:
        st.session_state.unified_chat_history = []
    if 'stored_visualizations' not in st.session_state:
        st.session_state.stored_visualizations = []
    if 'data_context' not in st.session_state:
        st.session_state.data_context = st.session_state.unified_viz_system.get_data_context(data)
    
    # Create tabs for different modes
    tab1, tab2, tab3 = st.tabs(["🤖 Multi-Agent Visualizations", "💬 Chat & Create", "📊 Visualization Gallery"])
    
    # Initialize chat history for sidebar
    if 'sidebar_chat_history' not in st.session_state:
        st.session_state.sidebar_chat_history = []
    
    with tab1:
        st.subheader("🤖 Multi-Agent Visualization System")
        st.write("Watch four specialized AI agents work together to create, review, and analyze your visualizations!")
        
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
        
        num_visualizations = st.slider("Select the number of visualizations", min_value=1, max_value=5, value=3)
        
        st.write("**Ready to see the agents in action?** Click the button below to start the multi-agent visualization process!")
        
        if st.button("🚀 Start Multi-Agent Visualization Process", type="primary"):
            with st.spinner("🤖 Multi-Agent System at Work..."):
                try:
                    # Agent 1: Generate visualization code
                    st.subheader("🤖 Agent 1: Code Generator")
                    st.write("**Task:** Analyzing data and generating visualization code...")
                    with st.spinner("Agent 1 is generating code..."):
                        generated_code = generate_visualization_code(data_sample, data_summary, num_visualizations)
                    
                    st.success("✅ Agent 1 completed: Code generated successfully!")
                    
                    with st.expander("🔍 Generated Code"):
                        st.code(generated_code, language='python')
                    
                    # Agent 2: Code review (syntax check)
                    st.subheader("🤖 Agent 2: Code Reviewer")
                    st.write("**Task:** Reviewing code for syntax errors and security issues...")
                    with st.spinner("Agent 2 is reviewing code..."):
                        review_feedback = st.session_state.unified_viz_system.review_code_syntax(generated_code)
                    
                    if "no syntax errors detected" not in review_feedback.lower():
                        st.error("❌ Agent 2 found issues: " + review_feedback)
                        return
                    else:
                        st.success("✅ Agent 2 completed: No syntax errors detected!")
                        st.info("🔍 Code Review Result: " + review_feedback)
                    
                    # Execute the generated code safely
                    st.subheader("📈 Generated Visualizations")
                    st.write("**Executing the reviewed code...**")
                    figs = execute_visualization_code(generated_code, filtered_data, num_visualizations)
                    for idx, fig in enumerate(figs, start=1):
                        st.plotly_chart(fig, use_container_width=True, key=f"ai_generated_chart_{idx}")
                    
                    # Agent 3: Enhanced interpretation of visualizations (data-centric)
                    st.subheader("🤖 Agent 3: Visualization Interpreter")
                    st.write("**Task:** Analyzing generated visualizations and extracting insights...")
                    with st.spinner("Agent 3 is analyzing visualizations..."):
                        interpretation = interpret_visualizations(figs, data_summary)
                    
                    st.success("✅ Agent 3 completed: Analysis finished!")
                    
                    with st.expander("💡 Visualization Interpretation"):
                        st.write(interpretation)
                    
                    # Enhanced Multi-Agent Discussion (displayed on main page)
                    st.subheader("🤖 Agent 4: Discussion Facilitator")
                    st.write("**Task:** Coordinating discussion among all agents and providing final insights...")
                    with st.spinner("Agent 4 is facilitating discussion..."):
                        discussion = st.session_state.unified_viz_system.agents_discussion(generated_code, review_feedback, interpretation)
                    
                    st.success("✅ Agent 4 completed: Discussion facilitated!")
                    
                    with st.expander("🤖 Multi-Agent Discussion"):
                        st.write(discussion)
                    
                    # Store visualization context
                    viz_context = {
                        "user_request": f"AI-generated {num_visualizations} visualizations",
                        "chart_type": "Multiple charts",
                        "insights": interpretation,
                        "discussion": discussion,
                        "timestamp": datetime.now(),
                        "figs": figs
                    }
                    st.session_state.stored_visualizations.append(viz_context)
                    
                    # Save context information for follow-up queries
                    context_info = f"Dataset Summary: {filtered_data.shape[0]} rows, {filtered_data.shape[1]} columns\n"
                    context_info += "Columns: " + ", ".join(filtered_data.columns)
                    context_info += "\nAgents Discussion Summary: " + discussion
                    st.session_state.context_info = context_info
                    
                    st.success("✅ Visualizations generated successfully!")
                    
                    # Enable sidebar chat after visualizations are created
                    st.session_state.show_sidebar_chat = True
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        

        

    
    with tab2:
        st.subheader("💬 Chat & Create Visualizations")
        st.write("Describe the chart you want and our AI agents will create it for you!")
        
        # User input
        user_input = st.text_input(
            "Describe the chart you want:",
            placeholder="Create a bar chart showing total sales by region...",
            key="chat_user_input"
        )
        
        if st.button("🚀 Start Multi-Agent Chart Creation", type="primary"):
            if user_input:
                with st.spinner("🤖 Multi-Agent System Working..."):
                    try:
                        # Create data summary for interpretation
                        data_summary = f"Dataset Summary:\nRows: {data.shape[0]}, Columns: {data.shape[1]}\nColumns: {', '.join(data.columns)}\nUser Request: {user_input}"
                        data_sample = data.head(5).to_csv(index=False)
                        
                        # Agent 1: Code Generator
                        st.subheader("🤖 Agent 1: Code Generator")
                        st.write("**Task:** Analyzing your request and generating visualization code...")
                        with st.spinner("Agent 1 is generating code..."):
                            # Use the same function as multi-agent tab but with user request
                            generated_code = generate_visualization_code(data_sample, data_summary, 1)
                        
                        st.success("✅ Agent 1 completed: Code generated successfully!")
                        
                        with st.expander("🔍 Generated Code"):
                            st.code(generated_code, language='python')
                        
                        # Agent 2: Code Reviewer
                        st.subheader("🔍 Agent 2: Code Reviewer")
                        st.write("**Task:** Reviewing code for syntax and security issues...")
                        with st.spinner("Agent 2 is reviewing code..."):
                            review_feedback = st.session_state.unified_viz_system.review_code_syntax(generated_code)
                        
                        if "no syntax errors detected" not in review_feedback.lower():
                            st.error("❌ Agent 2 found issues: " + review_feedback)
                            return
                        else:
                            st.success("✅ Agent 2 completed: No syntax errors detected!")
                            st.info("🔍 Code Review Result: " + review_feedback)
                        
                        # Execute the generated code safely
                        st.subheader("📈 Generated Visualization")
                        st.write("**Executing the reviewed code...**")
                        figs = execute_visualization_code(generated_code, data, 1)
                        
                        # Display the visualization
                        for idx, fig in enumerate(figs, start=1):
                            st.plotly_chart(fig, use_container_width=True, key=f"chat_generated_chart_{user_input}_{idx}")
                        
                        # Agent 3: Enhanced interpretation of visualizations
                        st.subheader("🤖 Agent 3: Visualization Interpreter")
                        st.write("**Task:** Analyzing generated visualization and extracting insights...")
                        with st.spinner("Agent 3 is analyzing visualization..."):
                            interpretation = interpret_visualizations(figs, data_summary)
                        
                        st.success("✅ Agent 3 completed: Analysis finished!")
                        
                        with st.expander("💡 Visualization Interpretation"):
                            st.write(interpretation)
                        
                        # Agent 4: Discussion Facilitator
                        st.subheader("🤖 Agent 4: Discussion Facilitator")
                        st.write("**Task:** Coordinating discussion among all agents and providing final insights...")
                        with st.spinner("Agent 4 is facilitating discussion..."):
                            discussion = st.session_state.unified_viz_system.agents_discussion(generated_code, review_feedback, interpretation)
                        
                        st.success("✅ Agent 4 completed: Discussion facilitated!")
                        
                        with st.expander("🤖 Multi-Agent Discussion"):
                            st.write(discussion)
                        
                        # Store visualization context
                        viz_context = {
                            "user_request": user_input,
                            "chart_type": "User-requested chart",
                            "insights": interpretation,
                            "discussion": discussion,
                            "timestamp": datetime.now(),
                            "figs": figs
                        }
                        st.session_state.stored_visualizations.append(viz_context)
                        
                        # Store in conversation history
                        st.session_state.unified_chat_history.append({
                            "type": "user",
                            "content": user_input,
                            "timestamp": datetime.now()
                        })
                        
                        st.session_state.unified_chat_history.append({
                            "type": "assistant",
                            "content": f"I've created a chart based on your request: '{user_input}'",
                            "chart": figs[0] if figs else None,
                            "insights": interpretation,
                            "discussion": discussion,
                            "timestamp": datetime.now()
                        })
                        
                        st.success("🎉 Chart created successfully with multi-agent collaboration!")
                        
                    except Exception as e:
                        st.error(f"Error in multi-agent process: {str(e)}")
                        st.session_state.unified_chat_history.append({
                            "type": "assistant",
                            "content": f"Sorry, I couldn't create that chart. Error: {str(e)}",
                            "timestamp": datetime.now()
                        })
            else:
                st.warning("Please enter a chart request.")
        
        # Show chat history
        if st.session_state.unified_chat_history:
            st.subheader("💬 Previous Requests")
            for i, message in enumerate(st.session_state.unified_chat_history[-5:]):  # Show last 5
                if message["type"] == "user":
                    st.write(f"**You:** {message['content']}")
                elif message["type"] == "assistant":
                    st.write(f"**AI:** {message['content']}")
                    if "chart" in message:
                        st.plotly_chart(message["chart"], use_container_width=True, key=f"history_chart_{i}")
                    if "insights" in message:
                        with st.expander(f"📊 Insights for request {i+1}"):
                            st.write(message["insights"])
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.unified_chat_history = []
            st.session_state.stored_visualizations = []
            st.session_state.unified_viz_system.memory.clear()
            st.success("History cleared!")
    
    with tab3:
        st.subheader("📊 Visualization Gallery")
        st.write("All your previously created visualizations are stored here.")
        
        if st.session_state.stored_visualizations:
            for i, viz in enumerate(st.session_state.stored_visualizations):
                with st.expander(f"Visualization {i+1}: {viz['user_request'][:50]}...", expanded=False):
                    st.write(f"**Request:** {viz['user_request']}")
                    st.write(f"**Type:** {viz['chart_type']}")
                    st.write(f"**Created:** {viz['timestamp'].strftime('%H:%M:%S')}")
                    
                    if 'figs' in viz:
                        for j, fig in enumerate(viz['figs']):
                            st.plotly_chart(fig, use_container_width=True, key=f"gallery_chart_{i}_{j}")
                    elif 'fig' in viz:
                        st.plotly_chart(viz['fig'], use_container_width=True, key=f"gallery_chart_{i}")
                    
                    if 'insights' in viz:
                        st.write("**Insights:**")
                        st.write(viz['insights'])
                    
                    if 'discussion' in viz:
                        st.write("**Agents Discussion:**")
                        st.write(viz['discussion'])
        else:
            st.info("No visualizations created yet. Try creating some in the other tabs!")
    
    # Sidebar for suggestions and chat
    with st.sidebar:
        st.header("💡 Suggestions")
        if st.button("Get Visualization Ideas"):
            suggestions = st.session_state.unified_viz_system.get_suggestions(
                st.session_state.data_context,
                st.session_state.unified_chat_history,
                st.session_state.stored_visualizations
            )
            st.session_state.suggestions = suggestions
        
        # Sidebar Chat Interface
        if hasattr(st.session_state, 'show_sidebar_chat') and st.session_state.show_sidebar_chat:
            st.write("---")
            st.header("💬 Ask About Your Visualizations")
            st.write("Ask questions about your data, visualizations, or request new charts!")
            
            # Show summary of available visualizations
            if st.session_state.stored_visualizations:
                st.subheader("📊 Available Visualizations")
                for i, viz in enumerate(st.session_state.stored_visualizations[-3:]):  # Show last 3
                    st.write(f"**{i+1}.** {viz['user_request'][:40]}...")
                    if 'insights' in viz:
                        st.caption(f"💡 {viz['insights'][:60]}...")
                st.write("---")
            
            # Display chat history
            if st.session_state.sidebar_chat_history:
                st.subheader("📝 Chat History")
                for message in st.session_state.sidebar_chat_history[-5:]:  # Show last 5 messages
                    if message["type"] == "user":
                        st.write(f"**You:** {message['content']}")
                    elif message["type"] == "assistant":
                        st.write(f"**AI:** {message['content']}")
                        if "sources" in message:
                            with st.expander("📚 Sources"):
                                for source in message["sources"]:
                                    st.write(f"**Chunk {source['chunk_id']}:**")
                                    st.write(source["content"])
                                    st.write("---")
            
            # Chat input
            sidebar_question = st.text_input(
                "Ask a question about your data or visualizations:",
                placeholder="What patterns do you see in the visualizations?",
                key="sidebar_chat_input"
            )
            
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("🔍 Ask Question", key="sidebar_ask_btn"):
                    if sidebar_question:
                        with st.spinner("Analyzing..."):
                            # Get context from stored visualizations and data
                            context = f"Dataset: {data.shape[0]} rows, {data.shape[1]} columns\n"
                            context += f"Columns: {', '.join(data.columns)}\n"
                            context += f"Stored visualizations: {len(st.session_state.stored_visualizations)}\n"
                            
                            if st.session_state.stored_visualizations:
                                context += "Recent visualizations:\n"
                                for viz in st.session_state.stored_visualizations[-3:]:  # Last 3 visualizations
                                    context += f"- {viz['user_request']}\n"
                                    if 'insights' in viz:
                                        context += f"  Insights: {viz['insights'][:100]}...\n"
                            
                            # Create a simple response based on context
                            response = generate_contextual_response(sidebar_question, context, st.session_state.stored_visualizations)
                            
                            # Store in chat history
                            st.session_state.sidebar_chat_history.append({
                                "type": "user",
                                "content": sidebar_question,
                                "timestamp": datetime.now()
                            })
                            
                            st.session_state.sidebar_chat_history.append({
                                "type": "assistant",
                                "content": response,
                                "timestamp": datetime.now()
                            })
                            
                            st.success("✅ Answer generated!")
                            
                            # Display the response immediately
                            st.write("**Latest Response:**")
                            st.write(response)
                    else:
                        st.warning("Please enter a question.")
            
            with col2:
                if st.button("🗑️ Clear Chat", key="sidebar_clear_btn"):
                    st.session_state.sidebar_chat_history = []
                    st.success("Chat history cleared!")
        
        if 'suggestions' in st.session_state:
            st.write("**Try these visualizations:**")
            st.write(st.session_state.suggestions)

# Import the necessary functions from visualizations.py
def filter_data(data):
    st.sidebar.header("Data Filtering Options")
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
    selected_columns = st.sidebar.multiselect("Select Categorical Columns for Filtering", options=categorical_columns)
    for col in selected_columns:
        unique_values = data[col].unique()
        selected_values = st.sidebar.multiselect(f"Select values for '{col}'", options=unique_values, default=unique_values)
        data = data[data[col].isin(selected_values)]
    return data

def generate_visualization_code(data_sample, dataset_stats, num_visualizations=3):
    """Generate visualization code using the existing function from visualizations.py"""
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

def execute_visualization_code(code, data, num_visualizations=3):
    """Execute the provided code after performing extra safety checks."""
    # Clean the code by removing import statements
    cleaned_code = ""
    for line in code.split('\n'):
        line = line.strip()
        # Skip import statements and empty lines
        if not line.startswith('import ') and not line.startswith('from ') and line:
            cleaned_code += line + '\n'
    
    disallowed_patterns = [
        r'\b__\b', r'\bopen\(', r'\beval\(', r'\bexec\(', r'\binput\(',
        r'\bcompile\(', r'\bos\b', r'\bsys\b', r'\bsubprocess\b', r'\bthreading\b',
        r'\bgetattr\(', r'\bsetattr\(', r'\bdelattr\(', r'\bglobals\(', r'\blocals\(',
        r'\bvars\(', r'\bhelp\(',
    ]
    for pattern in disallowed_patterns:
        if re.search(pattern, cleaned_code):
            raise ValueError(f"Disallowed pattern '{pattern}' found in code.")
    code_clean = cleaned_code.strip().replace('\r', '')
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

def interpret_visualizations(figs, data_summary):
    """Interpret visualizations using the existing function from visualizations.py"""
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

def generate_contextual_response(question, context, stored_visualizations):
    """Generate contextual responses based on stored visualizations and data context"""
    llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0.7)
    
    # Build context from stored visualizations
    viz_context = ""
    if stored_visualizations:
        viz_context = "Recent visualizations and insights:\n"
        for viz in stored_visualizations[-3:]:  # Last 3 visualizations
            viz_context += f"- Request: {viz['user_request']}\n"
            if 'insights' in viz:
                viz_context += f"  Insights: {viz['insights']}\n"
            if 'discussion' in viz:
                viz_context += f"  Discussion: {viz['discussion'][:200]}...\n"
    
    prompt_template = PromptTemplate(
        input_variables=["question", "context", "viz_context"],
        template="""
You are an expert data analyst assistant. Answer the user's question based on the provided context about their data and visualizations.

User Question: {question}

Data Context: {context}

Visualization Context: {viz_context}

Provide a helpful, informative response that:
1. Directly addresses the user's question
2. References relevant visualizations and insights when applicable
3. Suggests additional analysis if relevant
4. Uses the context provided to give specific, data-driven answers

Answer:
"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.run(
        question=question,
        context=context,
        viz_context=viz_context
    )
    return response.strip()