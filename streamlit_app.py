
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from scipy.stats import norm, linregress

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
# LOAD NASA GPM DATA
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
# DATE CONVERSION
# =====================================================

def decimal_year(dates):

    return (
        dates.dt.year
        + (dates.dt.dayofyear - 1) / 365.2425
    ).to_numpy(dtype=float)


# =====================================================
# SEASONAL MANN-KENDALL TEST
# =====================================================

def seasonal_mann_kendall(df):

    data = df[["Date", "Precipitation"]].dropna().copy()
    data["Month"] = data["Date"].dt.month

    total_s = 0
    total_variance = 0

    for _, group in data.groupby("Month"):

        values = group.sort_values("Date")[
            "Precipitation"
        ].to_numpy(dtype=float)

        n = len(values)

        if n < 2:
            continue

        month_s = 0

        for i in range(n - 1):
            month_s += np.sign(
                values[i + 1:] - values[i]
            ).sum()

        total_s += month_s

        _, counts = np.unique(
            values,
            return_counts=True
        )

        tie_term = np.sum(
            counts * (counts - 1) * (2 * counts + 5)
        )

        month_variance = (
            n * (n - 1) * (2 * n + 5) - tie_term
        ) / 18

        total_variance += month_variance

    if total_variance <= 0:

        return {
            "S": int(total_s),
            "Z": 0.0,
            "p_value": 1.0,
            "trend": "No statistically significant trend"
        }

    if total_s > 0:
        z_value = (
            total_s - 1
        ) / np.sqrt(total_variance)

    elif total_s < 0:
        z_value = (
            total_s + 1
        ) / np.sqrt(total_variance)

    else:
        z_value = 0.0

    p_value = 2 * norm.sf(abs(z_value))

    if p_value < ALPHA and z_value > 0:
        trend = "Increasing"

    elif p_value < ALPHA and z_value < 0:
        trend = "Decreasing"

    else:
        trend = "No statistically significant trend"

    return {
        "S": int(total_s),
        "Z": float(z_value),
        "p_value": float(p_value),
        "trend": trend
    }


# =====================================================
# SEASONAL SEN'S SLOPE
# =====================================================

def seasonal_sen_slope(df):

    data = df[["Date", "Precipitation"]].dropna().copy()

    data["Month"] = data["Date"].dt.month
    data["Year"] = decimal_year(data["Date"])

    slopes = []

    for _, group in data.groupby("Month"):

        group = group.sort_values("Year")

        x = group["Year"].to_numpy(dtype=float)
        y = group["Precipitation"].to_numpy(dtype=float)

        for i in range(len(y) - 1):

            time_difference = x[i + 1:] - x[i]
            value_difference = y[i + 1:] - y[i]

            valid = time_difference > 0

            if valid.any():

                month_slopes = (
                    value_difference[valid]
                    / time_difference[valid]
                )

                slopes.extend(month_slopes.tolist())

    if not slopes:
        return np.nan

    return float(np.median(slopes))


# =====================================================
# LAG-1 AUTOCORRELATION DIAGNOSTIC
# =====================================================

def seasonal_autocorrelation(df):

    data = df[["Date", "Precipitation"]].dropna().copy()
    data["Month"] = data["Date"].dt.month

    correlations = []

    for _, group in data.groupby("Month"):

        values = group.sort_values("Date")[
            "Precipitation"
        ].to_numpy(dtype=float)

        if len(values) < 4:
            continue

        first = values[:-1]
        second = values[1:]

        if np.std(first) == 0 or np.std(second) == 0:
            continue

        correlation = np.corrcoef(first, second)[0, 1]

        if np.isfinite(correlation):
            correlations.append(correlation)

    if not correlations:
        return np.nan

    return float(np.median(correlations))


# =====================================================
# ANNUAL MEAN REGRESSION
# =====================================================

def annual_regression(df):

    data = df.copy()

    data["Year"] = data["Date"].dt.year

    annual = (
        data.groupby("Year")["Precipitation"]
        .mean()
        .reset_index()
    )

    if len(annual) < 3:
        return annual, None

    regression = linregress(
        annual["Year"].to_numpy(dtype=float),
        annual["Precipitation"].to_numpy(dtype=float)
    )

    return annual, regression


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

st.divider()


# =====================================================
# MAIN DASHBOARD
# =====================================================

try:

    df = load_data()

    st.success("NASA GPM dataset loaded successfully.")

    # -------------------------------------------------
    # DATA QUALITY INFORMATION
    # -------------------------------------------------

    expected_dates = pd.date_range(
        start=df["Date"].min(),
        end=df["Date"].max(),
        freq="MS"
    )

    observed_dates = pd.DatetimeIndex(
        df["Date"].dt.to_period("M").dt.to_timestamp()
    )

    missing_months = len(
        expected_dates.difference(observed_dates)
    )

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

    if missing_months > 0:

        st.warning(
            f"Data quality notice: {missing_months} "
            "calendar months are missing from the dataset."
        )

    else:

        st.info(
            "No missing calendar months were detected "
            "between the first and last observations."
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

    if len(filtered_df) < 24:

        st.warning(
            "At least 24 monthly observations are recommended "
            "for this exploratory trend analysis."
        )

    else:

        # -------------------------------------------------
        # MONTHLY TIME SERIES
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

        st.divider()

        # -------------------------------------------------
        # STATISTICAL TREND ANALYSIS
        # -------------------------------------------------

        st.header("Statistical Trend Analysis")

        st.markdown("""
        Seasonal statistical methods are used to evaluate
        precipitation trends while accounting for the
        monthly seasonal structure.
        """)

        seasonal_result = seasonal_mann_kendall(
            filtered_df
        )

        sen_slope = seasonal_sen_slope(
            filtered_df
        )

        autocorrelation = seasonal_autocorrelation(
            filtered_df
        )

        annual, regression = annual_regression(
            filtered_df
        )

        # -------------------------------------------------
        # STATISTICAL METRICS
        # -------------------------------------------------

        col_a, col_b, col_c = st.columns(3)

        col_a.metric(
            "Seasonal Mann-Kendall p-value",
            f"{seasonal_result['p_value']:.3e}"
        )

        col_b.metric(
            "Seasonal Sen's Slope",
            f"{sen_slope:.6f} mm/hr/year"
            if np.isfinite(sen_slope)
            else "Not available"
        )

        col_c.metric(
            "Median Lag-1 Autocorrelation",
            f"{autocorrelation:.4f}"
            if np.isfinite(autocorrelation)
            else "Not available"
        )

        # -------------------------------------------------
        # TREND CLASSIFICATION
        # -------------------------------------------------

        st.subheader("Trend Classification")

        st.info(
            f"Seasonal Mann-Kendall result: "
            f"**{seasonal_result['trend']}**"
        )

        if seasonal_result["p_value"] < ALPHA:

            st.success(
                "The seasonal Mann-Kendall test indicates "
                "a statistically significant monotonic trend "
                "at the 5% significance level, under its "
                "independence assumption."
            )

        else:

            st.warning(
                "The test does not provide sufficient evidence "
                "of a monotonic trend at the 5% significance level."
            )

        # -------------------------------------------------
        # AUTOCORRELATION WARNING
        # -------------------------------------------------

        st.subheader("Temporal Dependence Diagnostic")

        if np.isfinite(autocorrelation):

            if abs(autocorrelation) >= 0.3:

                st.warning(
                    "Potential temporal dependence detected. "
                    "The p-value has not been adjusted for "
                    "serial autocorrelation and should be "
                    "interpreted cautiously."
                )

            else:

                st.info(
                    "The median lag-1 autocorrelation is below "
                    "the diagnostic threshold of 0.30. This "
                    "does not prove temporal independence."
                )

        else:

            st.warning(
                "Autocorrelation could not be estimated "
                "reliably for the selected period."
            )

        # -------------------------------------------------
        # ANNUAL REGRESSION
        # -------------------------------------------------

        st.divider()

        st.subheader("Annual Mean Precipitation Trend")

        if regression is not None:

            col_d, col_e, col_f = st.columns(3)

            col_d.metric(
                "Annual Regression R²",
                f"{regression.rvalue ** 2:.4f}"
            )

            col_e.metric(
                "Annual Regression Slope",
                f"{regression.slope:.6f} mm/hr/year"
            )

            col_f.metric(
                "Annual Regression p-value",
                f"{regression.pvalue:.3e}"
            )

            annual_fig = go.Figure()

            annual_fig.add_trace(
                go.Scatter(
                    x=annual["Year"],
                    y=annual["Precipitation"],
                    mode="lines+markers",
                    name="Annual mean precipitation"
                )
            )

            annual_fig.add_trace(
                go.Scatter(
                    x=annual["Year"],
                    y=regression.intercept
                    + regression.slope * annual["Year"],
                    mode="lines",
                    name="Annual linear regression"
                )
            )

            annual_fig.update_layout(
                title="Annual Mean Precipitation and Linear Trend",
                xaxis_title="Year",
                yaxis_title="Annual Mean Precipitation (mm/hr)",
                hovermode="x unified"
            )

            st.plotly_chart(
                annual_fig,
                use_container_width=True
            )

        else:

            st.info(
                "At least three annual observations are required "
                "for annual linear regression."
            )

        # -------------------------------------------------
        # STATISTICAL METHODOLOGY
        # -------------------------------------------------

        with st.expander("Statistical Methodology"):

            st.markdown(f"""
            **Seasonal Mann-Kendall Test**

            - S statistic: {seasonal_result['S']}
            - Z statistic: {seasonal_result['Z']:.5f}
            - p-value: {seasonal_result['p_value']:.8e}
            - Significance level: {ALPHA}
            - Method: comparisons between observations
              from the same calendar month across years.

            **Seasonal Sen's Slope**

            - Estimated slope: {sen_slope:.8f} mm/hr/year
              if available.
            - Method: median of pairwise slopes calculated
              within the same calendar month.

            **Temporal Dependence Diagnostic**

            - Median lag-1 autocorrelation:
              {autocorrelation:.5f}
            - Diagnostic threshold: 0.30
            - The p-value is not adjusted for serial
              autocorrelation.

            **Annual Linear Regression**

            - Annual observations: {len(annual)}
            - Slope: {regression.slope:.8f} mm/hr/year
              if available.
            - R-squared: {regression.rvalue ** 2:.6f}
              if available.

            **Interpretation and Limitations**

            The seasonal Mann-Kendall test and seasonal
            Sen's slope compare the same calendar months
            across different years.

            The autocorrelation value is a diagnostic only.
            It does not correct the Mann-Kendall variance
            or its p-value.

            Annual regression is based on annual mean
            precipitation rates and is presented as a
            complementary analysis.

            Results are exploratory and do not establish
            causation or regional rainfall patterns within Peru.
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

    st.error(
        f"Dataset loading error: {error}"
    )

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
