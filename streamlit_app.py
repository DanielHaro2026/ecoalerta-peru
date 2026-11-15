
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="ECOALERTA PERU",
    page_icon="🌎",
    layout="wide"
)

# =====================================================
# PROJECT CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "nasa_gpm_peru_1998_2025.csv"
)

FILL_VALUE = -9999.9

# =====================================================
# NASA DATA LOADING
# =====================================================

@st.cache_data
def load_nasa_data():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"NASA dataset not found: {DATA_FILE}"
        )

    # Skip the seven metadata rows from NASA Giovanni
    df = pd.read_csv(
        DATA_FILE,
        skiprows=7,
        skipinitialspace=True
    )

    df.columns = df.columns.str.strip()

    expected_columns = [
        "time",
        "mean_GPM_3IMERGM_07_precipitation"
    ]

    missing = [
        column for column in expected_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df[expected_columns].copy()

    df.columns = [
        "Date",
        "Precipitation"
    ]

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df["Precipitation"] = pd.to_numeric(
        df["Precipitation"],
        errors="coerce"
    )

    # Remove NASA fill values
    df.loc[
        df["Precipitation"] == FILL_VALUE,
        "Precipitation"
    ] = float("nan")

    df = (
        df.dropna()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError(
            "No valid precipitation records found."
        )

    return df


# =====================================================
# HEADER
# =====================================================

st.title("🌎 ECOALERTA PERU")

st.subheader("Earth System Trend Explorer")

st.markdown("""
**SPACE HACKERS UPRIT** | NASA Space Apps Challenge

An exploratory platform for analyzing precipitation
patterns across Peru using NASA GPM IMERG satellite data.
""")

st.divider()

# =====================================================
# DATA SOURCE
# =====================================================

st.header("NASA GPM IMERG Dataset")

st.markdown("""
- **Source:** NASA Global Precipitation Measurement
- **Product:** GPM_3IMERGM Version 07
- **Variable:** Merged satellite-gauge precipitation estimate – Final Run
- **Temporal resolution:** Monthly
- **Spatial selection:** Peru
- **Unit:** mm/hr
- **Period:** January 1998 – September 2025
""")

# =====================================================
# DASHBOARD
# =====================================================

try:

    df = load_nasa_data()

    st.success(
        "NASA GPM dataset loaded successfully."
    )

    # -------------------------------------------------
    # KEY INDICATORS
    # -------------------------------------------------

    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Valid Monthly Records",
        f"{len(df):,}"
    )

    col2.metric(
        "First Observation",
        df["Date"].min().strftime("%B %Y")
    )

    col3.metric(
        "Latest Observation",
        df["Date"].max().strftime("%B %Y")
    )

    st.divider()

    # -------------------------------------------------
    # DATE FILTER
    # -------------------------------------------------

    st.header("Precipitation Trend Explorer")

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()

    selected_dates = st.date_input(
        "Select analysis period",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if (
        isinstance(selected_dates, (tuple, list))
        and len(selected_dates) == 2
    ):

        start_date, end_date = selected_dates

        filtered_df = df[
            (df["Date"].dt.date >= start_date)
            & (df["Date"].dt.date <= end_date)
        ].copy()

    else:
        filtered_df = df.copy()

    if filtered_df.empty:

        st.warning(
            "No observations available "
            "for the selected period."
        )

    else:

        # -------------------------------------------------
        # TIME SERIES CHART
        # -------------------------------------------------

        st.subheader("Monthly Precipitation Time Series")

        fig = px.line(
            filtered_df,
            x="Date",
            y="Precipitation",
            title="NASA GPM IMERG Precipitation - Peru",
            labels={
                "Date": "Date",
                "Precipitation": "Precipitation (mm/hr)"
            }
        )

        fig.update_layout(
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Precipitation (mm/hr)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # -------------------------------------------------
        # DESCRIPTIVE STATISTICS
        # -------------------------------------------------

        st.header("Descriptive Statistics")

        col_a, col_b, col_c = st.columns(3)

        col_a.metric(
            "Mean Precipitation",
            f"{filtered_df['Precipitation'].mean():.4f} mm/hr"
        )

        col_b.metric(
            "Maximum Precipitation",
            f"{filtered_df['Precipitation'].max():.4f} mm/hr"
        )

        col_c.metric(
            "Minimum Precipitation",
            f"{filtered_df['Precipitation'].min():.4f} mm/hr"
        )

        # -------------------------------------------------
        # DATA TABLE
        # -------------------------------------------------

        with st.expander("View NASA GPM Data Records"):

            st.dataframe(
                filtered_df,
                use_container_width=True,
                hide_index=True
            )

        # -------------------------------------------------
        # DOWNLOAD FILTERED DATA
        # -------------------------------------------------

        csv_download = filtered_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download filtered data (CSV)",
            data=csv_download,
            file_name="ecoalerta_peru_precipitation.csv",
            mime="text/csv"
        )

except Exception as error:

    st.error(
        f"Dataset loading error: {error}"
    )

    st.info("""
    Verify that the NASA CSV file exists in the data folder
    and contains the expected Giovanni column names.
    """)

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption("""
Data source: NASA GPM IMERG Version 07.

Independent research prototype developed by
SPACE HACKERS UPRIT.
""")
