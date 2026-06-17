import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(
    page_title="Universal Data Analyzer",
    layout="wide"
)

st.title("📊 Universal Data Analyzer")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV, Excel, or Parquet file",
    type=["csv", "xlsx", "xls", "parquet"]
)

if uploaded_file:

    # -------------------------------
    # Load Data
    # -------------------------------
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    elif uploaded_file.name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)

    elif uploaded_file.name.endswith(".parquet"):
        df = pd.read_parquet(uploaded_file)

    st.success("Data Loaded Successfully")

    # -------------------------------
    # Dataset Overview
    # -------------------------------
    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.dataframe(df.head())

    # -------------------------------
    # Data Types
    # -------------------------------
    st.header("Column Information")

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing": df.isnull().sum(),
        "Unique Values": df.nunique()
    })

    st.dataframe(info_df)

    # -------------------------------
    # Missing Values
    # -------------------------------
    st.header("Missing Value Analysis")

    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": df.isnull().sum(),
        "Missing %": (df.isnull().sum() / len(df) * 100).round(2)
    })

    st.dataframe(missing_df)

    fig = px.bar(
        missing_df,
        x="Column",
        y="Missing %",
        title="Missing Value Percentage"
    )
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # Statistical Summary
    # -------------------------------
    st.header("Statistical Summary")

    st.dataframe(df.describe(include="all").T)

    # -------------------------------
    # Filtering
    # -------------------------------
    st.header("Interactive Filter")

    filter_col = st.selectbox(
        "Select Column",
        df.columns
    )

    if df[filter_col].dtype == "object":

        values = st.multiselect(
            "Choose Values",
            df[filter_col].dropna().unique()
        )

        if values:
            filtered_df = df[df[filter_col].isin(values)]
        else:
            filtered_df = df

    else:

        min_val = float(df[filter_col].min())
        max_val = float(df[filter_col].max())

        selected_range = st.slider(
            "Range",
            min_val,
            max_val,
            (min_val, max_val)
        )

        filtered_df = df[
            (df[filter_col] >= selected_range[0]) &
            (df[filter_col] <= selected_range[1])
        ]

    st.write("Filtered Data")
    st.dataframe(filtered_df)

    # -------------------------------
    # Correlation Heatmap
    # -------------------------------
    numeric_df = df.select_dtypes(include=np.number)

    if len(numeric_df.columns) > 1:

        st.header("Correlation Heatmap")

        corr = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            ax=ax
        )

        st.pyplot(fig)

    # -------------------------------
    # Histogram
    # -------------------------------
    if len(numeric_df.columns) > 0:

        st.header("Histogram")

        hist_col = st.selectbox(
            "Select Numeric Column",
            numeric_df.columns,
            key="hist"
        )

        fig = px.histogram(
            df,
            x=hist_col,
            nbins=30,
            title=f"Distribution of {hist_col}"
        )

        st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # Scatter Plot
    # -------------------------------
    if len(numeric_df.columns) >= 2:

        st.header("Scatter Plot")

        x_col = st.selectbox(
            "X Axis",
            numeric_df.columns,
            key="scatter_x"
        )

        y_col = st.selectbox(
            "Y Axis",
            numeric_df.columns,
            index=1,
            key="scatter_y"
        )

        color_col = st.selectbox(
            "Color By (Optional)",
            ["None"] + list(df.columns)
        )

        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=None if color_col == "None" else color_col,
            title=f"{x_col} vs {y_col}"
        )

        st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # Box Plot
    # -------------------------------
    if len(numeric_df.columns) > 0:

        st.header("Box Plot")

        box_col = st.selectbox(
            "Select Column",
            numeric_df.columns,
            key="box"
        )

        fig = px.box(
            df,
            y=box_col,
            title=f"Outlier Analysis - {box_col}"
        )

        st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # Download Data
    # -------------------------------
    st.header("Download")

    csv = filtered_df.to_csv(index=False)

    st.download_button(
        label="Download Filtered Data",
        data=csv,
        file_name="filtered_data.csv",
        mime="text/csv"
    )

else:
    st.info("Upload a dataset to begin analysis.")