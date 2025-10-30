import os
import requests
from dotenv import load_dotenv

load_dotenv()

HISTORICAL_API_URL = os.getenv("HISTORICAL_API_URL")
GEOCODING_API_URL = os.getenv("GEOCODING_API_URL")
WEATHER_API_URL = os.getenv("WEATHER_API_URL")

def get_geocoding_data(city):
    try:
        query_params = {"name": city, "limit": 1, "language": "fr", "format": "json"}
        response = requests.get(GEOCODING_API_URL, params=query_params)
        response.raise_for_status()
        data = response.json()
        if data and "results" in data and len(data["results"]) > 0:
            result = data["results"][0]
            filtered_data = {
                "latitude": result.get("latitude"),
                "longitude": result.get("longitude"),
                "country_code": result.get("country_code"),
                "timezone": result.get("timezone")
            }
            return filtered_data
        return None
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la connexion à l'API : {e}")
        return None

def get_hourly_weather_data(geolocalisation, start_date, end_date):
    try:
        query_params = {
            "latitude": geolocalisation["latitude"], 
            "longitude": geolocalisation["longitude"], 
            "hourly": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,wind_gusts_10m,precipitation,apparent_temperature,cloud_cover,wind_direction_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,sunshine_duration",
            "timezone": geolocalisation["timezone"],
            "start_date": start_date,
            "end_date": end_date
        }
        response = requests.get(WEATHER_API_URL, params=query_params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la connexion à l'API : {e}")
        return None

def get_historical_weather_data(geolocalisation, start_date, end_date):
    try:
        query_params = {
            "latitude": geolocalisation["latitude"],
            "longitude": geolocalisation["longitude"],
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,wind_gusts_10m,precipitation,apparent_temperature,cloud_cover,wind_direction_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,sunshine_duration"
        }
        response = requests.get(HISTORICAL_API_URL, params=query_params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la connexion à l'API : {e}")
        return None
