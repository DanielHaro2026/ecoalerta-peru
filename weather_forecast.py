
import requests
import pandas as pd

API_URL = "https://api.open-meteo.com/v1/forecast"

# Coordenadas de referencia de las capitales departamentales
DEPARTMENTS = {
    "Amazonas": (-6.23, -77.87),
    "Ancash": (-9.53, -77.53),
    "Apurimac": (-13.64, -72.88),
    "Arequipa": (-16.40, -71.54),
    "Ayacucho": (-13.16, -74.22),
    "Cajamarca": (-7.16, -78.51),
    "Callao": (-12.06, -77.14),
    "Cusco": (-13.52, -71.97),
    "Huancavelica": (-12.79, -74.97),
    "Huanuco": (-9.93, -76.24),
    "Ica": (-14.07, -75.73),
    "Junin": (-11.16, -75.99),
    "La Libertad": (-8.11, -79.03),
    "Lambayeque": (-6.77, -79.84),
    "Lima": (-12.05, -77.04),
    "Loreto": (-3.75, -73.25),
    "Madre de Dios": (-12.59, -69.19),
    "Moquegua": (-17.19, -70.94),
    "Pasco": (-10.68, -76.26),
    "Piura": (-5.19, -80.63),
    "Puno": (-15.84, -70.02),
    "San Martin": (-6.49, -76.37),
    "Tacna": (-18.01, -70.25),
    "Tumbes": (-3.57, -80.45),
    "Ucayali": (-8.38, -74.53),
}


def get_department_forecast(department):
    """Obtiene el pronóstico meteorológico de una capital departamental."""

    if department not in DEPARTMENTS:
        raise ValueError(f"Departamento no registrado: {department}")

    latitude, longitude = DEPARTMENTS[department]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": [
            "precipitation_sum",
            "precipitation_probability_max",
            "temperature_2m_max",
            "temperature_2m_min",
        ],
        "hourly": [
            "precipitation",
            "precipitation_probability",
        ],
        "forecast_days": 7,
        "timezone": "America/Lima",
    }

    response = requests.get(API_URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    daily = pd.DataFrame(data["daily"])

    return daily


def get_all_department_forecasts():
    """Consulta los pronósticos de los 25 ámbitos departamentales."""

    results = []

    for department in DEPARTMENTS:
        try:
            forecast = get_department_forecast(department)

            tomorrow = forecast.iloc[1]

            results.append({
                "Departamento": department,
                "Fecha": tomorrow["time"],
                "Lluvia_mm": tomorrow["precipitation_sum"],
                "Probabilidad_lluvia": tomorrow[
                    "precipitation_probability_max"
                ],
                "Temperatura_max": tomorrow["temperature_2m_max"],
                "Temperatura_min": tomorrow["temperature_2m_min"],
                "Estado": "Disponible",
            })

        except (requests.RequestException, KeyError, IndexError, ValueError) as error:
            results.append({
                "Departamento": department,
                "Fecha": None,
                "Lluvia_mm": None,
                "Probabilidad_lluvia": None,
                "Temperatura_max": None,
                "Temperatura_min": None,
                "Estado": f"No disponible: {error}",
            })

    return pd.DataFrame(results)
