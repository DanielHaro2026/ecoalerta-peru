
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from scipy.stats import theilslopes, linregress, norm

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="ECOALERTA PERU",
    page_icon="🌎",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / "nasa_gpm_peru_1998_2025.csv"

FILL_VALUE = -9999.9
ALPHA = 0.05

# =====================================================
# LOAD NASA DATA
# =====================================================

@st.cache_data
def load_data():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE,
        skiprows=7,
        skipinitialspace=True
    )

    df.columns = df.columns.str.strip()

    date_col = "time"
    value_col = "mean_GPM_3IMERGM_07_precipitation"

    if date_col not in df.columns or value_col not in df.columns:
        raise ValueError(
            f"Unexpected CSV columns: {df.columns.tolist()}"
        )

    df = df[[date_col, value_col]].copy()

    df.columns = ["Date", "Precipitation"]

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df["Precipitation"] = pd.to_numeric(
        df["Precipitation"],
        errors="coerce"
    )

    df.loc[
        df["Precipitation"] == FILL_VALUE,
        "Precipitation"
    ] = np.nan

    df = (
        df.dropna()
        .sort_values("Date")
        .drop_duplicates(subset="Date")
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError("No valid NASA records found.")

    return df


# =====================================================
# SEASONAL MANN-KENDALL TEST
# =====================================================

def seasonal_mann_kendall(values, months):

    values = np.asarray(values, dtype=float)
    months = np.asarray(months)

    S_total = 0
    variance_total = 0

    for month in np.unique(months):

        sample = values[months == month]
        n = len(sample)

        if n < 2:
            continue

        S_month = 0

        for i in range(n - 1):
            S_month += np.sum(
                np.sign(sample[i + 1:] - sample[i])
            )

        S_total += S_month

        _, counts = np.unique(
            sample,
            return_counts=True
        )

        tie_term = np.sum(
            counts * (counts - 1) * (2 * counts + 5)
        )

        variance_total += (
            n * (n - 1) * (2 * n + 5) - tie_term
        ) / 18

    if variance_total <= 0:
        return {
            "S": S_total,
            "Z": 0.0,
            "p_value": 1.0,
            "trend": "No detectable trend"
        }

    if S_total > 0:
        Z = (S_total - 1) / np.sqrt(variance_total)
    elif S_total < 0:
        Z = (S_total + 1) / np.sqrt(variance_total)
    else:
        Z = 0.0

    p_value = 2 * norm.sf(abs(Z))

    if p_value < ALPHA and Z > 0:
        trend = "Increasing"
    elif p_value < ALPHA and Z < 0:
        trend = "Decreasing"
    else:
        trend = "No statistically significant trend"

    return {
        "S": S_total,
        "Z": Z,
        "p_value": p_value,
        "trend": trend
    }


# =====================================================
# SEN'S SLOPE
# =====================================================

def calculate_sen_slope(df):

    x = (
        df["Date"].dt.year
        + (df["Date"].dt.dayofyear - 1) / 365.2425
    ).to_numpy()

    y = df["Precipitation"].to_numpy()

    if len(y) < 2:
        return np.nan, np.nan, np.nan

    result = theilslopes(y, x)

    return result.slope, result.low_slope, result.high_slope


# =====================================================
# LINEAR REGRESSION
# =====================================================

def calculate_regression(df):

    x = (
        df["Date"].dt.year
        + (df["Date"].dt.dayofyear - 1) / 365.2425
    ).to_numpy()

    y = df["Precipitation"].to_numpy()

    result = linregress(x, y)

    return result


# =====================================================
# HEADER
# =====================================================

st.title("🌎 ECOALERTA PERU")

st.subheader("Earth System Trend Explorer")

st.markdown("""
**SPACE HACKERS UPRIT** | NASA Space Apps Challenge

Exploring precipitation patterns across Peru
using NASA GPM IMERG satellite observations.
""")

st.divider()

# =====================================================
# DATASET INFORMATION
# =====================================================

st.header("NASA GPM IMERG Dataset")

st.markdown("""
- **Source:** NASA Global Precipitation Measurement
- **Product:** GPM_3IMERGM Version 07
- **Variable:** Merged satellite-gauge precipitation estimate – Final Run
- **Temporal resolution:** Monthly
- **Spatial selection:** Peru
- **Measurement unit:** mm/hr
""")

# =====================================================
# MAIN DASHBOARD
# =====================================================

try:

    df = load_data()

    st.success("NASA GPM dataset loaded successfully.")

    col1, col2, col3 = st.columns(3)

    col1.metric("Valid Monthly Records", f"{len(df):,}")
    col2.metric("First Observation", df["Date"].min().strftime("%B %Y"))
    col3.metric("Latest Observation", df["Date"].max().strftime("%B %Y"))

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

    if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:

        start_date, end_date = selected_dates

        filtered_df = df[
            (df["Date"].dt.date >= start_date)
            & (df["Date"].dt.date <= end_date)
        ].copy()

    else:
        filtered_df = df.copy()

    if len(filtered_df) < 24:

        st.warning(
            "At least 24 monthly observations are recommended "
            "for this exploratory trend analysis."
        )

    else:

        # -------------------------------------------------
        # TIME SERIES
        # -------------------------------------------------

        st.subheader("Monthly Precipitation Time Series")

        fig = px.line(
            filtered_df,
            x="Date",
            y="Precipitation",
            title="NASA GPM IMERG Monthly Precipitation",
            labels={
                "Date": "Date",
                "Precipitation": "Precipitation (mm/hr)"
            }
        )

        fig.update_layout(
            hovermode="x unified"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # -------------------------------------------------
        # DESCRIPTIVE STATISTICS
        # -------------------------------------------------

        st.header("Descriptive Statistics")

        a, b, c = st.columns(3)

        a.metric(
            "Mean",
            f"{filtered_df['Precipitation'].mean():.4f} mm/hr"
        )

        b.metric(
            "Maximum",
            f"{filtered_df['Precipitation'].max():.4f} mm/hr"
        )

        c.metric(
            "Minimum",
            f"{filtered_df['Precipitation'].min():.4f} mm/hr"
        )

        # -------------------------------------------------
        # STATISTICAL ANALYSIS
        # -------------------------------------------------

        st.divider()
        st.header("Statistical Trend Analysis")

        st.markdown("""
        The following tests evaluate the direction and magnitude
        of precipitation changes over the selected period.
        """)

        seasonal_result = seasonal_mann_kendall(
            filtered_df["Precipitation"].to_numpy(),
            filtered_df["Date"].dt.month.to_numpy()
        )

        sen_slope, sen_low, sen_high = calculate_sen_slope(
            filtered_df
        )

        regression = calculate_regression(filtered_df)

        col_a, col_b, col_c = st.columns(3)

        col_a.metric(
            "Mann-Kendall p-value",
            f"{seasonal_result['p_value']:.5f}"
        )

        col_b.metric(
            "Sen's Slope",
            f"{sen_slope:.6f} mm/hr/year"
        )

        col_c.metric(
            "Linear Regression R²",
            f"{regression.rvalue ** 2:.4f}"
        )

        st.subheader("Trend Classification")

        st.info(
            f"Seasonal Mann-Kendall result: "
            f"**{seasonal_result['trend']}**"
        )

        if seasonal_result["p_value"] < ALPHA:
            st.success(
                "The test indicates a statistically significant "
                "monotonic trend at the 5% significance level."
            )
        else:
            st.warning(
                "The test does not provide sufficient evidence "
                "of a monotonic trend at the 5% significance level."
            )

        # -------------------------------------------------
        # REGRESSION CHART
        # -------------------------------------------------

        regression_line = (
            regression.intercept
            + regression.slope * (
                filtered_df["Date"].dt.year
                + (
                    filtered_df["Date"].dt.dayofyear - 1
                ) / 365.2425
            )
        )

        trend_fig = go.Figure()

        trend_fig.add_trace(
            go.Scatter(
                x=filtered_df["Date"],
                y=filtered_df["Precipitation"],
                mode="lines",
                name="Observed precipitation"
            )
        )

        trend_fig.add_trace(
            go.Scatter(
                x=filtered_df["Date"],
                y=regression_line,
                mode="lines",
                name="Linear regression"
            )
        )

        trend_fig.update_layout(
            title="Observed Precipitation and Linear Trend",
            xaxis_title="Date",
            yaxis_title="Precipitation (mm/hr)",
            hovermode="x unified"
        )

        st.plotly_chart(
            trend_fig,
            use_container_width=True
        )

        # -------------------------------------------------
        # METHOD NOTES
        # -------------------------------------------------

        with st.expander("Statistical Methodology"):

            st.markdown(f"""
            **Seasonal Mann-Kendall**

            - S statistic: {seasonal_result['S']}
            - Z statistic: {seasonal_result['Z']:.4f}
            - p-value: {seasonal_result['p_value']:.6f}
            - Significance level: {ALPHA}

            **Sen's Slope**

            - Estimated slope: {sen_slope:.8f} mm/hr/year
            - Lower confidence limit: {sen_low:.8f}
            - Upper confidence limit: {sen_high:.8f}

            **Linear Regression**

            - Slope: {regression.slope:.8f}
            - R-squared: {regression.rvalue ** 2:.6f}
            - Regression p-value: {regression.pvalue:.6f}

            Results are exploratory. Serial correlation,
            data homogeneity and other assumptions require
            additional scientific evaluation.
            """)

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
        # DOWNLOAD
        # -------------------------------------------------

        st.download_button(
            "Download filtered data (CSV)",
            filtered_df.to_csv(index=False).encode("utf-8"),
            file_name="ecoalerta_peru_precipitation.csv",
            mime="text/csv"
        )

except Exception as error:

    st.error(f"Dataset loading error: {error}")

    st.info(
        "Check the NASA CSV file and its expected column names."
    )

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption("""
Data source: NASA GPM IMERG Version 07.

Independent research prototype developed by
SPACE HACKERS UPRIT.
""")
