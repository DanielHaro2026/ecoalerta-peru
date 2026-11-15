
import streamlit as st

st.set_page_config(
    page_title="ECOALERTA PERÚ",
    page_icon="🌎",
    layout="wide"
)

st.title("🌎 ECOALERTA PERÚ")
st.subheader("Earth System Trend Explorer")

st.markdown("""
### Detectando tendencias de precipitación en el Perú

Proyecto desarrollado por **SPACE HACKERS UPRIT**
para NASA Space Apps Challenge.

Utilizamos datos científicos de NASA GPM IMERG
para estudiar los patrones de precipitación en el territorio peruano.
""")

st.info(
    "Prototipo en desarrollo. "
    "La integración y el análisis de datos NASA se incorporarán "
    "en las siguientes etapas."
)

st.divider()

st.subheader("Explorador de tendencias")

st.write(
    "Aquí presentaremos los resultados del análisis "
    "de precipitación por regiones del Perú."
)
