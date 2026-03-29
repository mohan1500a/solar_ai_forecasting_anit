import requests
import pandas as pd
from datetime import datetime
import time
import logging

# --- Setup Logging ---
logger = logging.getLogger("WeatherAPI")

def fetch_open_meteo_forecast(lat=9.575390, lon=77.675835, max_retries=3):
    """
    Fetches 7-day hourly forecast from Open-Meteo with robust error handling.
    Includes retry logic and timeout for production-grade reliability.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,cloud_cover,shortwave_radiation,wind_speed_10m,pressure_msl",
        "timezone": "auto",
        "forecast_days": 7
    }
    
    for attempt in range(max_retries):
        try:
            # 10 second timeout for responsiveness
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                hourly_data = data['hourly']
                
                df = pd.DataFrame(hourly_data)
                df['time'] = pd.to_datetime(df['time'])
                
                # Rename for consistent internal pipeline (mapped to original project feature names)
                df = df.rename(columns={
                    "temperature_2m": "temperature_2m (°C)",
                    "relative_humidity_2m": "relative_humidity",
                    "cloud_cover": "cloudcover",
                    "shortwave_radiation": "shortwave_radiation (W/m²)",
                    "wind_speed_10m": "windspeed_10m",
                    "pressure_msl": "pressure_msl"
                })
                
                return df
                
            elif response.status_code == 429:
                logger.warning("Weather API Rate Limit hit. Retrying in 2s...")
                time.sleep(2)
            else:
                raise Exception(f"Weather API Error {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error on attempt {attempt+1}/{max_retries}: {e}")
            if attempt == max_retries - 1:
                raise e
            time.sleep(1) # Backoff
            
    return None

if __name__ == "__main__":
    try:
        # Quick health check test
        forecast = fetch_open_meteo_forecast()
        if forecast is not None:
            print(f"Successfully fetched forecast for {forecast['time'].iloc[0]} to {forecast['time'].iloc[-1]}")
            print(forecast[['time', 'shortwave_radiation (W/m²)', 'cloudcover']].head())
    except Exception as e:
        print(f"API Research Test Error: {e}")
