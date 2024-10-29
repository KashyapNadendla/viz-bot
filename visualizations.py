import streamlit as st
from plotly import express as px
from utils import get_visualization_suggestions  # Function for LLM suggestions

def visualization_section():
    st.header("Interactive Visualizations")

    # Ensure data is available in session state
    if 'data' not in st.session_state:
        st.error("Please upload a dataset first.")
        return

    data = st.session_state['data']  # Reference the dataset in session state

    st.subheader("Manual Visualizations")

    # Allow users to select plot type
    plot_types = [
        "Scatter Plot", "Line Plot", "Bar Chart", "Histogram",
        "Box Plot", "Violin Plot", "Pie Chart", "Heatmap"
    ]
    plot_type = st.selectbox("Select Plot Type", plot_types)

    # Collect column options based on data types
    all_columns = data.columns.tolist()
    numeric_cols = data.select_dtypes(include=['float', 'int']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

    # Initialize variables
    x_axis = y_axis = color = size = None

    # Adjust options based on plot type
    if plot_type in ["Scatter Plot", "Line Plot"]:
        x_axis = st.selectbox("Select X-axis", options=all_columns)
        y_axis = st.selectbox("Select Y-axis", options=all_columns)
        if st.checkbox("Add Size Dimension"):
            size = st.selectbox("Select Size Column", options=numeric_cols)
    elif plot_type in ["Bar Chart", "Box Plot", "Violin Plot"]:
        x_axis = st.selectbox("Select X-axis", options=all_columns)
        y_axis = st.selectbox("Select Y-axis", options=all_columns)
    elif plot_type == "Histogram":
        x_axis = st.selectbox("Select Column", options=all_columns)
    elif plot_type == "Pie Chart":
        x_axis = st.selectbox("Select Names Column", options=all_columns)
        y_axis = st.selectbox("Select Values Column (optional)", options=numeric_cols + [None])
    elif plot_type == "Heatmap":
        st.write("Select columns for Heatmap")
        x_axis = st.multiselect("X-axis Categories", options=categorical_cols)
        y_axis = st.multiselect("Y-axis Categories", options=categorical_cols)
        z_value = st.selectbox("Values (numeric)", options=numeric_cols)

    # Color encoding option
    color_option = st.checkbox("Add Color Differentiation")
    if color_option:
        color = st.selectbox("Select Column for Color", options=all_columns)

    # Generate plot based on user selections
    if st.button("Generate Plot"):
        fig = None
        try:
            if plot_type == "Scatter Plot":
                fig = px.scatter(data, x=x_axis, y=y_axis, color=color, size=size, hover_data=all_columns)
            elif plot_type == "Line Plot":
                fig = px.line(data, x=x_axis, y=y_axis, color=color)
            elif plot_type == "Bar Chart":
                fig = px.bar(data, x=x_axis, y=y_axis, color=color)
            elif plot_type == "Histogram":
                fig = px.histogram(data, x=x_axis, color=color)
            elif plot_type == "Box Plot":
                fig = px.box(data, x=x_axis, y=y_axis, color=color)
            elif plot_type == "Violin Plot":
                fig = px.violin(data, x=x_axis, y=y_axis, color=color, box=True, points='all')
            elif plot_type == "Pie Chart":
                fig = px.pie(data, names=x_axis, values=y_axis, color=color)
            elif plot_type == "Heatmap":
                if x_axis and y_axis and z_value:
                    pivot_table = data.pivot_table(index=y_axis, columns=x_axis, values=z_value, aggfunc='mean')
                    fig = px.imshow(pivot_table, aspect='auto', color_continuous_scale='Viridis')
                else:
                    st.error("Please select appropriate columns for Heatmap.")
            else:
                st.error("Unsupported plot type selected.")

            if fig:
                st.plotly_chart(fig)
        except Exception as e:
            st.error(f"An error occurred: {e}")

    st.subheader("AI-Powered Visualization Suggestions")
    if st.button("Get Visualization Suggestions"):
        with st.spinner("Generating suggestions..."):
            suggestions = get_visualization_suggestions(data)
        st.write(suggestions)
