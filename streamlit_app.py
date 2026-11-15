
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from scipy.stats import norm, linregress


# =========================================================
# 1. CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ECOALERTA PERU | Earth System Trend Explorer",
    page_icon="🌎",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / "nasa_gpm_peru_1998_2025.csv"

FILL_VALUE = -9999.9
ALPHA = 0.05
BOOTSTRAP_REPS = 1999
RANDOM_SEED = 42


# =========================================================
# 2. DATA LOADING AND CLEANING
# =========================================================

@st.cache_data
def load_data():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"NASA dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE,
        skiprows=7,
        skipinitialspace=True
    )

    df.columns = df.columns.str.strip()

    required_columns = [
        "time",
        "mean_GPM_3IMERGM_07_precipitation"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing CSV columns: {missing_columns}"
        )

    df = df[required_columns].copy()

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
        raise ValueError(
            "The dataset contains no valid observations."
        )

    return df


# =========================================================
# 3. DATE UTILITIES
# =========================================================

def decimal_year(dates):

    return (
        dates.dt.year
        + (dates.dt.dayofyear - 1) / 365.2425
    ).to_numpy(dtype=float)


# =========================================================
# 4. SEASONAL SEN'S SLOPE
# =========================================================

def seasonal_sen_slope(df):

    data = df[
        ["Date", "Precipitation"]
    ].dropna().copy()

    data["Month"] = data["Date"].dt.month
    data["YearDecimal"] = decimal_year(data["Date"])

    slopes = []

    for _, group in data.groupby("Month"):

        group = group.sort_values("YearDecimal")

        x = group["YearDecimal"].to_numpy(dtype=float)
        y = group["Precipitation"].to_numpy(dtype=float)

        for i in range(len(y) - 1):

            time_difference = x[i + 1:] - x[i]
            value_difference = y[i + 1:] - y[i]

            valid = time_difference > 0

            if valid.any():

                pairwise_slopes = (
                    value_difference[valid]
                    / time_difference[valid]
                )

                slopes.extend(
                    pairwise_slopes.tolist()
                )

    if not slopes:
        return np.nan

    return float(np.median(slopes))


# =========================================================
# 5. SEASONAL MANN-KENDALL S STATISTIC
# =========================================================

def seasonal_s_statistic(matrix):

    total_s = 0

    for month_index in range(matrix.shape[1]):

        values = matrix[:, month_index]

        values = values[
            np.isfinite(values)
        ]

        for i in range(len(values) - 1):

            total_s += np.sign(
                values[i + 1:] - values[i]
            ).sum()

    return int(total_s)


# =========================================================
# 6. BLOCK BOOTSTRAP SEASONAL MANN-KENDALL
# =========================================================

def seasonal_mann_kendall_bootstrap(
    df,
    bootstrap_reps=BOOTSTRAP_REPS
):

    data = df[
        ["Date", "Precipitation"]
    ].dropna().copy()

    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month

    matrix_df = data.pivot_table(
        index="Year",
        columns="Month",
        values="Precipitation",
        aggfunc="mean"
    )

    matrix_df = matrix_df.reindex(
        columns=range(1, 13)
    )

    # Use complete calendar years for annual block resampling.
    matrix_df = matrix_df.dropna(
        axis=0,
        how="any"
    )

    matrix = matrix_df.to_numpy(dtype=float)
    years = matrix_df.index.to_numpy(dtype=float)

    n_years = len(years)

    if n_years < 8:

        return {
            "S": np.nan,
            "p_value": np.nan,
            "trend": "Insufficient complete years",
            "block_p_values": {},
            "complete_years": n_years,
            "bootstrap_reps": bootstrap_reps
        }

    observed_s = seasonal_s_statistic(matrix)

    complete_data = data[
        data["Year"].isin(matrix_df.index)
    ].copy()

    estimated_slope = seasonal_sen_slope(
        complete_data
    )

    if not np.isfinite(estimated_slope):
        estimated_slope = 0.0

    # Remove the estimated trend to construct a
    # detrended residual series under the null hypothesis.
    centered_years = years - years.mean()

    residuals = matrix - (
        estimated_slope
        * centered_years[:, np.newaxis]
    )

    # Center each calendar month independently.
    residuals = residuals - residuals.mean(
        axis=0,
        keepdims=True
    )

    rng = np.random.default_rng(RANDOM_SEED)

    block_lengths = [2, 3, 4]
    block_p_values = {}

    for block_length in block_lengths:

        exceedances = 0

        blocks_needed = int(
            np.ceil(n_years / block_length)
        )

        for _ in range(bootstrap_reps):

            starts = rng.integers(
                low=0,
                high=n_years,
                size=blocks_needed
            )

            indices = []

            for start in starts:

                block_indices = (
                    start + np.arange(block_length)
                ) % n_years

                indices.extend(
                    block_indices.tolist()
                )

            indices = np.asarray(
                indices[:n_years],
                dtype=int
            )

            bootstrap_matrix = residuals[indices, :]

            bootstrap_s = seasonal_s_statistic(
                bootstrap_matrix
            )

            if abs(bootstrap_s) >= abs(observed_s):
                exceedances += 1

        empirical_p = (
            exceedances + 1
        ) / (
            bootstrap_reps + 1
        )

        block_p_values[block_length] = float(
            empirical_p
        )

    # Conservative sensitivity value across block lengths.
    final_p_value = max(
        block_p_values.values()
    )

    if final_p_value < ALPHA and observed_s > 0:

        trend = "Increasing"

    elif final_p_value < ALPHA and observed_s < 0:

        trend = "Decreasing"

    else:

        trend = "No statistically significant trend"

    return {
        "S": observed_s,
        "p_value": float(final_p_value),
        "trend": trend,
        "block_p_values": block_p_values,
        "complete_years": n_years,
        "bootstrap_reps": bootstrap_reps
    }


# =========================================================
# 7. TEMPORAL AUTOCORRELATION DIAGNOSTIC
# =========================================================

def seasonal_autocorrelation(df):

    data = df[
        ["Date", "Precipitation"]
    ].dropna().copy()

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

        if (
            np.std(first) == 0
            or np.std(second) == 0
        ):
            continue

        correlation = np.corrcoef(
            first,
            second
        )[0, 1]

        if np.isfinite(correlation):
            correlations.append(correlation)

    if not correlations:
        return np.nan

    return float(
        np.median(correlations)
    )


# =========================================================
# 8. ANNUAL LINEAR REGRESSION
# =========================================================

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


# =========================================================
# 9. APPLICATION HEADER
# =========================================================

st.title("🌎 ECOALERTA PERU")

st.subheader(
    "Earth System Trend Explorer"
)

st.markdown("""
**SPACE HACKERS UPRIT** | NASA Space Apps Challenge

Exploring precipitation patterns across Peru
using NASA GPM IMERG satellite observations.
""")

st.divider()


# =========================================================
# 10. DATASET INFORMATION
# =========================================================

st.header("NASA GPM IMERG Dataset")

st.markdown("""
- **Source:** NASA Global Precipitation Measurement
- **Product:** GPM_3IMERGM Version 07
- **Variable:** Merged satellite-gauge precipitation estimate – Final Run
- **Temporal resolution:** Monthly
- **Spatial selection:** Peru
- **Measurement unit:** mm/hr
- **Observation period:** January 1998 – September 2025
""")

st.divider()


# =========================================================
# 11. MAIN DASHBOARD
# =========================================================

try:

    df = load_data()

    st.success(
        "NASA GPM dataset loaded successfully."
    )

    # -----------------------------------------------------
    # DATA QUALITY
    # -----------------------------------------------------

    expected_dates = pd.date_range(
        start=df["Date"].min().to_period("M").to_timestamp(),
        end=df["Date"].max().to_period("M").to_timestamp(),
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
            f"{missing_months} missing calendar months "
            "were detected in the observation period."
        )

    else:

        st.info(
            "No missing calendar months were detected "
            "between the first and last observations."
        )

    st.divider()

    # -----------------------------------------------------
    # DATE FILTER
    # -----------------------------------------------------

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
            "for exploratory trend analysis."
        )

    else:

        # -------------------------------------------------
        # MONTHLY TIME SERIES
        # -------------------------------------------------

        st.subheader(
            "Monthly Precipitation Time Series"
        )

        monthly_fig = px.line(
            filtered_df,
            x="Date",
            y="Precipitation",
            title="NASA GPM IMERG Monthly Precipitation",
            labels={
                "Date": "Date",
                "Precipitation": "Precipitation (mm/hr)"
            }
        )

        monthly_fig.update_layout(
            hovermode="x unified"
        )

        st.plotly_chart(
            monthly_fig,
            use_container_width=True
        )

        # -------------------------------------------------
        # DESCRIPTIVE STATISTICS
        # -------------------------------------------------

        st.header("Descriptive Statistics")

        stat1, stat2, stat3 = st.columns(3)

        stat1.metric(
            "Mean",
            f"{filtered_df['Precipitation'].mean():.4f} mm/hr"
        )

        stat2.metric(
            "Maximum",
            f"{filtered_df['Precipitation'].max():.4f} mm/hr"
        )

        stat3.metric(
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

        seasonal_result = seasonal_mann_kendall_bootstrap(
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
            "Block-bootstrap Seasonal MK p-value",
            (
                f"{seasonal_result['p_value']:.3e}"
                if np.isfinite(seasonal_result["p_value"])
                else "Not available"
            )
        )

        col_b.metric(
            "Seasonal Sen's Slope",
            (
                f"{sen_slope:.6f} mm/hr/year"
                if np.isfinite(sen_slope)
                else "Not available"
            )
        )

        col_c.metric(
            "Median Lag-1 Autocorrelation",
            (
                f"{autocorrelation:.4f}"
                if np.isfinite(autocorrelation)
                else "Not available"
            )
        )

        # -------------------------------------------------
        # TREND CLASSIFICATION
        # -------------------------------------------------

        st.subheader("Trend Classification")

        st.info(
            "Block-bootstrap Seasonal Mann-Kendall result: "
            f"**{seasonal_result['trend']}**"
        )

        if seasonal_result["trend"] == "Decreasing":

            st.success(
                "The sensitivity analysis indicates a decreasing "
                "monotonic trend at the 5% significance level."
            )

        elif seasonal_result["trend"] == "Increasing":

            st.success(
                "The sensitivity analysis indicates an increasing "
                "monotonic trend at the 5% significance level."
            )

        elif seasonal_result["trend"] == "No statistically significant trend":

            st.warning(
                "The analysis does not provide sufficient evidence "
                "of a monotonic trend at the 5% significance level."
            )

        else:

            st.warning(
                "There are not enough complete calendar years "
                "to perform the block-bootstrap analysis."
            )

        # -------------------------------------------------
        # BOOTSTRAP SENSITIVITY
        # -------------------------------------------------

        st.subheader(
            "Block Bootstrap Sensitivity Analysis"
        )

        st.markdown("""
        The analysis evaluates three block lengths.
        The largest empirical p-value is reported as a
        conservative sensitivity result.
        """)

        if seasonal_result["block_p_values"]:

            bootstrap_table = pd.DataFrame(
                [
                    {
                        "Block Length (years)": block_length,
                        "Empirical p-value": p_value
                    }
                    for block_length, p_value
                    in seasonal_result["block_p_values"].items()
                ]
            )

            st.dataframe(
                bootstrap_table,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                f"Complete years used: "
                f"{seasonal_result['complete_years']} | "
                f"Bootstrap repetitions per block length: "
                f"{seasonal_result['bootstrap_reps']} | "
                f"Random seed: {RANDOM_SEED}"
            )

        # -------------------------------------------------
        # AUTOCORRELATION DIAGNOSTIC
        # -------------------------------------------------

        st.subheader(
            "Temporal Dependence Diagnostic"
        )

        if np.isfinite(autocorrelation):

            if abs(autocorrelation) >= 0.3:

                st.warning(
                    "Potential temporal dependence detected. "
                    "Interpret trend estimates cautiously."
                )

            else:

                st.info(
                    "The median lag-1 autocorrelation is below "
                    "the diagnostic threshold of 0.30. "
                    "This does not prove temporal independence."
                )

        else:

            st.warning(
                "Autocorrelation could not be estimated reliably."
            )

        # -------------------------------------------------
        # ANNUAL REGRESSION
        # -------------------------------------------------

        st.divider()

        st.subheader(
            "Annual Mean Precipitation Trend"
        )

        if regression is not None:

            reg1, reg2, reg3 = st.columns(3)

            reg1.metric(
                "Annual Regression R²",
                f"{regression.rvalue ** 2:.4f}"
            )

            reg2.metric(
                "Annual Regression Slope",
                f"{regression.slope:.6f} mm/hr/year"
            )

            reg3.metric(
                "Annual Regression p-value",
                f"{regression.pvalue:.3e}"
            )

            annual_fig = go.Figure()

            annual_fig.add_trace(
                go.Scatter(
                    x=annual["Year"],
                    y=annual["Precipitation"],
                    mode="lines+markers",
                    name="Annual Mean Precipitation"
                )
            )

            annual_fig.add_trace(
                go.Scatter(
                    x=annual["Year"],
                    y=(
                        regression.intercept
                        + regression.slope * annual["Year"]
                    ),
                    mode="lines",
                    name="Linear Regression"
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
        # METHODOLOGY AND LIMITATIONS
        # -------------------------------------------------

        with st.expander(
            "Statistical Methodology and Limitations"
        ):

            st.markdown(f"""
            **Seasonal Mann-Kendall with Block Bootstrap**

            - S statistic: {seasonal_result['S']}
            - Reported empirical p-value: {
                seasonal_result['p_value']:.8e
                if np.isfinite(seasonal_result['p_value'])
                else 'Not available'
            }
            - Complete calendar years: {
                seasonal_result['complete_years']
            }
            - Bootstrap repetitions per block length: {
                seasonal_result['bootstrap_reps']
            }
            - Block lengths: 2, 3 and 4 years.
            - Reported p-value: maximum across evaluated block lengths.

            **Seasonal Sen's Slope**

            - Estimated slope: {
                f"{sen_slope:.8f} mm/hr/year"
                if np.isfinite(sen_slope)
                else "Not available"
            }
            - Pairwise slopes are calculated only between
              observations from the same calendar month.

            **Temporal Dependence**

            - Median lag-1 autocorrelation: {
                f"{autocorrelation:.5f}"
                if np.isfinite(autocorrelation)
                else "Not available"
            }
            - This diagnostic does not establish independence.

            **Annual Linear Regression**

            - Annual observations: {len(annual)}
            - Regression R-squared: {
                f"{regression.rvalue ** 2:.6f}"
                if regression is not None
                else "Not available"
            }

            **Methodological Limitations**

            The block-bootstrap procedure is an approximate
            sensitivity analysis. It uses detrended annual
            residual vectors and circular moving blocks.

            It requires complete calendar years and may be
            sensitive to the selected block lengths and the
            estimated trend removed before resampling.

            The reported p-value is not an exact correction
            for every form of serial dependence.

            Annual regression is a complementary descriptive
            analysis and does not replace the seasonal test.

            The dataset represents a spatially averaged
            precipitation estimate for the selected Peru
            polygon. It does not establish trends in every
            region of the country.

            These results are exploratory and do not establish
            causation.
            """)

        # -------------------------------------------------
        # DATA TABLE
        # -------------------------------------------------

        with st.expander(
            "View NASA GPM Data Records"
        ):

            st.dataframe(
                filtered_df,
                use_container_width=True,
                hide_index=True
            )

        # -------------------------------------------------
        # CSV DOWNLOAD
        # -------------------------------------------------

        st.download_button(
            label="Download filtered data (CSV)",
            data=filtered_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="ecoalerta_peru_precipitation.csv",
            mime="text/csv"
        )


except Exception as error:

    st.error(
        f"Application error: {error}"
    )

    st.info(
        "Verify the dataset location, CSV structure "
        "and required columns."
    )


# =========================================================
# 12. FOOTER
# =========================================================

st.divider()

st.caption("""
Data source: NASA GPM IMERG Version 07.

Independent research prototype developed by
SPACE HACKERS UPRIT.
""")
