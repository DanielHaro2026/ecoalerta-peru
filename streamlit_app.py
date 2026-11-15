
import streamlit as st

st.set_page_config(
    page_title="ECOALERTA PERU",
    page_icon="🌎",
    layout="wide"
)

st.title("🌎 ECOALERTA PERU")

st.subheader("Earth System Trend Explorer")

st.header("Detecting Precipitation Trends Across Peru")

st.markdown("""
Developed by **SPACE HACKERS UPRIT**
for the **NASA Space Apps Challenge**.

We use NASA GPM IMERG satellite precipitation data
to study rainfall patterns across Peruvian territory.
""")

st.info(
    "Development prototype. NASA data integration "
    "and scientific analysis are currently in progress."
)

st.divider()

st.header("Precipitation Trend Explorer")

st.write(
    "Explore precipitation patterns and trends "
    "across the different regions of Peru."
)
