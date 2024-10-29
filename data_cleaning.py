import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder


def data_cleaning_section():
    # Use the data from session state
    data = st.session_state.get('data', None)

    if data is None:
        st.warning("Please upload and load data before proceeding to this section.")
        return

    st.header("Data Cleaning and Preprocessing")

    # Missing Values
    with st.expander("Handle Missing Values"):
        st.write("**Missing Values in Each Column:**")
        st.dataframe(data.isnull().sum())
        missing_value_strategy = st.selectbox(
            "Select a strategy to handle missing values",
            ["Do nothing", "Drop missing values", "Impute with mean", "Impute with median", "Impute with mode"]
        )
        if st.button("Apply Missing Value Strategy"):
            if missing_value_strategy == "Drop missing values":
                st.session_state.data.dropna(inplace=True)
                st.success("Dropped missing values.")
            elif missing_value_strategy in ["Impute with mean", "Impute with median", "Impute with mode"]:
                numeric_cols = data.select_dtypes(include=['float', 'int']).columns
                if missing_value_strategy == "Impute with mean":
                    st.session_state.data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
                elif missing_value_strategy == "Impute with median":
                    st.session_state.data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
                elif missing_value_strategy == "Impute with mode":
                    st.session_state.data = data.fillna(data.mode().iloc[0])
                st.success(f"Imputed missing values with {missing_value_strategy.lower()}.")

    # Handling Duplicates
    with st.expander("Handle Duplicates"):
        st.write(f"**Number of duplicate rows:** {data.duplicated().sum()}")
        if st.button("Remove Duplicates"):
            st.session_state.data.drop_duplicates(inplace=True)
            st.success("Duplicates removed.")

    # Data Type Conversion
    with st.expander("Convert Data Types"):
        st.write("**Current Data Types:**")
        st.dataframe(data.dtypes)
        cols_to_convert = st.multiselect("Select columns to convert to numeric", options=data.columns)
        if st.button("Convert to Numeric"):
            for col in cols_to_convert:
                st.session_state.data[col] = pd.to_numeric(data[col], errors='coerce')
            st.success("Converted selected columns to numeric.")

    # Encoding Features
    with st.expander("Feature Encoding"):
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        encode_cols = st.multiselect("Select categorical columns to encode", options=categorical_cols)
        encoding_type = st.selectbox("Select Encoding Type", ["One-Hot Encoding", "Label Encoding"])
        if st.button("Encode Features"):
            if encoding_type == "One-Hot Encoding":
                st.session_state.data = pd.get_dummies(data, columns=encode_cols)
                st.success("Applied One-Hot Encoding.")
            else:
                le = LabelEncoder()
                for col in encode_cols:
                    st.session_state.data[col] = le.fit_transform(data[col].astype(str))
                st.success("Applied Label Encoding.")

    # Scaling and Normalization
    with st.expander("Scaling and Normalization"):
        numeric_cols = data.select_dtypes(include=['float', 'int']).columns.tolist()
        scale_cols = st.multiselect("Select columns to scale", options=numeric_cols)
        scaling_method = st.selectbox("Select Scaling Method", ["StandardScaler", "MinMaxScaler", "RobustScaler"])
        if st.button("Scale Features"):
            scaler = None
            if scaling_method == "StandardScaler":
                scaler = StandardScaler()
            elif scaling_method == "MinMaxScaler":
                scaler = MinMaxScaler()
            elif scaling_method == "RobustScaler":
                scaler = RobustScaler()
            if scaler and scale_cols:
                st.session_state.data[scale_cols] = scaler.fit_transform(data[scale_cols])
                st.success(f"Scaled selected features using {scaling_method}.")
            else:
                st.warning("Please select at least one column to scale.")

    # Handling Outliers
    with st.expander("Handle Outliers"):
        numeric_cols = data.select_dtypes(include=['float', 'int']).columns.tolist()
        outlier_cols = st.multiselect("Select columns to handle outliers", options=numeric_cols)
        outlier_method = st.selectbox("Select Outlier Handling Method", ["Remove", "Cap"])
        if st.button("Handle Outliers"):
            for col in outlier_cols:
                q1 = data[col].quantile(0.25)
                q3 = data[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                if outlier_method == "Remove":
                    st.session_state.data = st.session_state.data[(data[col] >= lower) & (data[col] <= upper)]
                elif outlier_method == "Cap":
                    st.session_state.data[col] = data[col].clip(lower, upper)
            st.success(f"Outliers handled using {outlier_method} method.")

    # Datetime Feature Extraction
    with st.expander("Datetime Feature Extraction"):
        datetime_cols = data.select_dtypes(include=['datetime64[ns]', 'object']).columns.tolist()
        datetime_cols = [col for col in datetime_cols if pd.api.types.is_datetime64_any_dtype(data[col])]
        if datetime_cols:
            datetime_col = st.selectbox("Select datetime column", options=datetime_cols)
            if st.button("Extract Datetime Features"):
                st.session_state.data['year'] = data[datetime_col].dt.year
                st.session_state.data['month'] = data[datetime_col].dt.month
                st.session_state.data['day'] = data[datetime_col].dt.day
                st.session_state.data['hour'] = data[datetime_col].dt.hour
                st.success("Extracted year, month, day, and hour from datetime column.")
        else:
            st.write("No datetime columns available for feature extraction.")

    # Reset Data
    if st.button("Reset Data to Original"):
        st.session_state.data = st.session_state.original_data.copy()
        st.success("Data reset to original.")

    st.write("### Preprocessed Data")
    st.dataframe(st.session_state.data)
