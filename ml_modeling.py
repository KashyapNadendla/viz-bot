# ml_modeling.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import numpy as np
import plotly.express as px
import json

from utils import analyze_with_llm  # Import the function from utils.py

def ml_modeling_section():
    if 'data' not in st.session_state:
        st.warning("Please upload and preprocess your data before proceeding to this section.")
        return

    data = st.session_state['data']  # Use session state data    
    st.header("Machine Learning Modeling and Evaluation")

    # Select target and features
    target = st.selectbox("Select Target Variable", options=data.columns)
    features = st.multiselect(
        "Select Feature Variables", options=[col for col in data.columns if col != target]
    )

    if not target or not features:
        st.warning("Please select a target variable and at least one feature.")
        return

    # Detect and confirm problem type
    if data[target].dtype in ['int64', 'float64']:
        problem_type_default = 0  # Regression
    else:
        problem_type_default = 1  # Classification

    problem_type = st.radio(
        "Select Problem Type", ["Regression", "Classification"], index=problem_type_default
    )

    # Model options based on problem type
    model_options = {
        "Regression": ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"],
        "Classification": [
            "Logistic Regression",
            "Decision Tree Classifier",
            "Random Forest Classifier",
            "Support Vector Machine",
        ],
    }
    selected_model = st.selectbox("Select Model", model_options[problem_type])

    # Option to use cross-validation
    use_cross_validation = st.checkbox("Use Cross-Validation")
    cv_folds = (
        st.number_input(
            "Number of Folds (k)", min_value=2, max_value=20, value=5, step=1
        )
        if use_cross_validation
        else None
    )

    # Gather user selections to send to LLM later
    user_selections = {
        "Target Variable": target,
        "Features": features,
        "Problem Type": problem_type,
        "Selected Model": selected_model,
        "Cross-Validation": use_cross_validation,
        "Cross-Validation Folds": int(cv_folds) if use_cross_validation else "N/A"
    }

    if st.button("Train and Evaluate Model"):
        X = pd.get_dummies(data[features], drop_first=True)
        y = data[target]

        if problem_type == "Classification" and y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Initialize the model
        model = None
        if problem_type == "Regression":
            model_dict = {
                "Linear Regression": LinearRegression(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
            }
            model = model_dict[selected_model]
        elif problem_type == "Classification":
            if selected_model == "Support Vector Machine":
                model = make_pipeline(StandardScaler(), SVC(probability=True))
            else:
                model_dict = {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "Decision Tree Classifier": DecisionTreeClassifier(),
                    "Random Forest Classifier": RandomForestClassifier(),
                }
                model = model_dict[selected_model]

        # Initialize model_results dictionary
        model_results = {}

        if use_cross_validation:
            st.write("**Cross-Validation Results:**")
            if problem_type == "Regression":
                scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
                cv_results = cross_validate(
                    model, X, y, cv=int(cv_folds), scoring=scoring
                )
                model_results['Average R²'] = np.mean(cv_results['test_r2'])
                model_results['Average MSE'] = -np.mean(cv_results['test_neg_mean_squared_error'])
                model_results['Average MAE'] = -np.mean(cv_results['test_neg_mean_absolute_error'])
                st.write(f"Average R² Score: {model_results['Average R²']:.2f}")
                st.write(f"Average MSE: {model_results['Average MSE']:.2f}")
                st.write(f"Average MAE: {model_results['Average MAE']:.2f}")
            else:
                scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
                cv_results = cross_validate(
                    model, X, y, cv=int(cv_folds), scoring=scoring
                )
                model_results['Average Accuracy'] = np.mean(cv_results['test_accuracy'])
                model_results['Average Precision'] = np.mean(cv_results['test_precision_weighted'])
                model_results['Average Recall'] = np.mean(cv_results['test_recall_weighted'])
                model_results['Average F1 Score'] = np.mean(cv_results['test_f1_weighted'])
                st.write(f"Average Accuracy: {model_results['Average Accuracy']:.2f}")
                st.write(f"Average Precision: {model_results['Average Precision']:.2f}")
                st.write(f"Average Recall: {model_results['Average Recall']:.2f}")
                st.write(f"Average F1 Score: {model_results['Average F1 Score']:.2f}")
        else:
            # Train-test split and model evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            if problem_type == "Regression":
                # Evaluation metrics
                mse = mean_squared_error(y_test, y_pred_test)
                mae = mean_absolute_error(y_test, y_pred_test)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred_test)
                model_results['MSE'] = mse
                model_results['MAE'] = mae
                model_results['RMSE'] = rmse
                model_results['R²'] = r2

                st.write("**Regression Metrics on Test Set:**")
                st.write(f"Mean Squared Error (MSE): {mse:.2f}")
                st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
                st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
                st.write(f"R-squared Score (R²): {r2:.2f}")

                # Visualizations
                st.subheader("Visualizations")
                tab1, tab2 = st.tabs(["Actual vs Predicted", "Residuals"])

                with tab1:
                    st.write("**Test Set:**")
                    fig_test = px.scatter(
                        x=y_test, y=y_pred_test,
                        labels={'x': 'Actual', 'y': 'Predicted'},
                        title='Actual vs Predicted Values (Test Set)'
                    )
                    fig_test.add_shape(
                        type='line', x0=y_test.min(), y0=y_test.min(),
                        x1=y_test.max(), y1=y_test.max(),
                        line=dict(color='red', dash='dash')
                    )
                    st.plotly_chart(fig_test, key="test_plot")

                    st.write("**Training Set:**")
                    fig_train = px.scatter(
                        x=y_train, y=y_pred_train,
                        labels={'x': 'Actual', 'y': 'Predicted'},
                        title='Actual vs Predicted Values (Training Set)'
                    )
                    fig_train.add_shape(
                        type='line', x0=y_train.min(), y0=y_train.min(),
                        x1=y_train.max(), y1=y_train.max(),
                        line=dict(color='red', dash='dash')
                    )
                    st.plotly_chart(fig_train, key="train_plot")

                with tab2:
                    residuals = y_test - y_pred_test
                    fig_residuals = px.histogram(
                        residuals, nbins=30,
                        title='Residuals Distribution (Test Set)'
                    )
                    st.plotly_chart(fig_residuals, key="residuals_plot")

                # Feature Importance
                if selected_model in ["Decision Tree Regressor", "Random Forest Regressor"]:
                    feature_importances = model.feature_importances_
                    importance_df = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': feature_importances
                    }).sort_values(by='Importance', ascending=False)
                    model_results['Feature Importances'] = importance_df.to_dict(orient='records')
                    st.subheader("Feature Importance")
                    fig_importance = px.bar(
                        importance_df, x='Feature', y='Importance',
                        title='Feature Importance'
                    )
                    st.plotly_chart(fig_importance, key="importance_plot_regression")

            elif problem_type == "Classification":
                # Evaluation metrics
                report = classification_report(y_test, y_pred_test, output_dict=True)
                model_results['Classification Report'] = report
                st.write("**Classification Report on Test Set:**")
                st.text(classification_report(y_test, y_pred_test))

                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred_test)
                model_results['Confusion Matrix'] = cm.tolist()
                st.write("Confusion Matrix:")
                fig_cm = px.imshow(
                    cm, text_auto=True, color_continuous_scale='Blues',
                    labels=dict(x="Predicted", y="Actual"),
                    x=sorted(np.unique(y_test)), y=sorted(np.unique(y_test)),
                    title="Confusion Matrix"
                )
                st.plotly_chart(fig_cm, key="confusion_matrix")

                # ROC Curve and AUC (for binary classification)
                if len(np.unique(y_test)) == 2:
                    if hasattr(model, "predict_proba"):
                        y_proba = model.predict_proba(X_test)[:, 1]
                    else:
                        y_proba = model.decision_function(X_test)
                        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
                    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
                    auc_score = roc_auc_score(y_test, y_proba)
                    model_results['AUC Score'] = auc_score
                    st.write(f"AUC Score: {auc_score:.2f}")
                    fig_roc = px.area(
                        x=fpr, y=tpr,
                        title=f'ROC Curve (AUC = {auc_score:.2f})',
                        labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'}
                    )
                    fig_roc.add_shape(
                        type='line', x0=0, y0=0, x1=1, y1=1,
                        line=dict(color='red', dash='dash')
                    )
                    st.plotly_chart(fig_roc, key="roc_curve")

                    # Precision-Recall Curve
                    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_proba)
                    fig_pr = px.area(
                        x=recall, y=precision,
                        title='Precision-Recall Curve',
                        labels={'x': 'Recall', 'y': 'Precision'}
                    )
                    st.plotly_chart(fig_pr, key="precision_recall_curve")

                # Feature Importance
                if selected_model in ["Decision Tree Classifier", "Random Forest Classifier"]:
                    feature_importances = model.feature_importances_
                    importance_df = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': feature_importances
                    }).sort_values(by='Importance', ascending=False)
                    model_results['Feature Importances'] = importance_df.to_dict(orient='records')
                    st.subheader("Feature Importance")
                    fig_importance = px.bar(
                        importance_df, x='Feature', y='Importance',
                        title='Feature Importance'
                    )
                    st.plotly_chart(fig_importance, key="importance_plot_classification")

        # Convert data sample to JSON string
        data_sample = data.head(5).to_json(orient='records')

        # Prepare model results for LLM (convert numpy data types to native Python types)
        model_results_clean = json.loads(json.dumps(model_results, default=lambda o: o.tolist() if hasattr(o, 'tolist') else o))

        # Call the analyze_with_llm function
        llm_analysis = analyze_with_llm(
            data_sample=data_sample,
            user_selections=user_selections,
            model_results=model_results_clean
        )

        # Display LLM analysis
        st.subheader("AI Interpretation and Recommendations")
        st.write(llm_analysis)
