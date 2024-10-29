# import streamlit as st
# from utils import analyze_data_with_llm

# def llm_analysis_section(data):
#     st.header("LLM Analysis and Suggestions")

#     st.write("Select the type of analysis you want the LLM to perform:")

#     # Create buttons for different types of LLM analyses
#     if st.button("Analyze Data"):
#         with st.spinner("Analyzing data..."):
#             response = analyze_data_with_llm(data, prompt_type="analysis")
#         st.write("### Data Analysis and Suggestions")
#         st.write(response)

#     if st.button("Suggest Models"):
#         with st.spinner("Generating model suggestions..."):
#             response = analyze_data_with_llm(data, prompt_type="model_suggestions")
#         st.write("### Model Suggestions")
#         st.write(response)

#     if st.button("Visualization Suggestions"):
#         with st.spinner("Generating visualization suggestions..."):
#             response = analyze_data_with_llm(data, prompt_type="visualization_suggestions")
#         st.write("### Visualization Suggestions")
#         st.write(response)

#     if st.button("Feature Engineering Suggestions"):
#         with st.spinner("Generating feature engineering suggestions..."):
#             response = analyze_data_with_llm(data, prompt_type="feature_engineering")
#         st.write("### Feature Engineering Suggestions")
#         st.write(response)

import streamlit as st
from utils import analyze_data_with_llm, get_visualization_suggestions

def llm_analysis_section():
    if 'data' not in st.session_state:
        st.warning("Please upload and preprocess your data before proceeding to this section.")
        return

    data = st.session_state['data']  # Use session state data
    st.header("LLM Analysis and Suggestions Chat")

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for entry in st.session_state.chat_history:
        if entry["type"] == "user":
            st.write(f"**You**: {entry['message']}")
        elif entry["type"] == "bot":
            st.write(f"**LLM**: {entry['message']}")

    # User selects the type of analysis
    st.write("Select the type of analysis you want the LLM to perform:")

    if st.button("Analyze Data"):
        with st.spinner("Analyzing data..."):
            response = analyze_data_with_llm(data, prompt_type="analysis")
        st.session_state.chat_history.append({"type": "bot", "message": response})
        st.write("### Data Analysis and Suggestions")
        st.write(response)

    if st.button("Suggest Models"):
        with st.spinner("Generating model suggestions..."):
            response = analyze_data_with_llm(data, prompt_type="model_suggestions")
        st.session_state.chat_history.append({"type": "bot", "message": response})
        st.write("### Model Suggestions")
        st.write(response)

    if st.button("Visualization Suggestions"):
        with st.spinner("Generating visualization suggestions..."):
            response = get_visualization_suggestions(data)
        st.session_state.chat_history.append({"type": "bot", "message": response})
        st.write("### Visualization Suggestions")
        st.write(response)

    if st.button("Feature Engineering Suggestions"):
        with st.spinner("Generating feature engineering suggestions..."):
            response = analyze_data_with_llm(data, prompt_type="feature_engineering")
        st.session_state.chat_history.append({"type": "bot", "message": response})
        st.write("### Feature Engineering Suggestions")
        st.write(response)

    # Follow-up question input
    follow_up_question = st.text_input("Ask a follow-up question:")
    if st.button("Submit Question"):
        if follow_up_question:
            st.session_state.chat_history.append({"type": "user", "message": follow_up_question})

            with st.spinner("Processing your question..."):
                # Use the last bot response as context for the new question
                last_response = st.session_state.chat_history[-1]["message"]
                follow_up_response = analyze_data_with_llm(data, prompt_type="custom", last_response=last_response, question=follow_up_question)
            
            st.session_state.chat_history.append({"type": "bot", "message": follow_up_response})
            st.write("### Response")
            st.write(follow_up_response)
