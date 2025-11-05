import sys
import types
import datetime as _dt
import pandas as pd
import numpy as np
import pytest

# --- Alias modules for the imports (do refactor chưa đồng bộ)
import core.interfaces as _ifaces
import data.transformer as _transformer
import services.weather_service as _weather_service
import services.analytics.forecasting as _forecasting
import services.analytics.pca as _pca
import data.transforms as _transforms
import data.data_cleaning as _data_cleaning


# --- Fixtures / helpers general

@pytest.fixture
def sample_daily_json():
    """Giống schema daily của Open-Meteo (rút gọn)."""
    return {
        "daily": {
            "time": ["2024-10-01", "2024-10-02", "2024-10-03"],
            "weathercode": [1, 3, 61],
            "temperature_2m_mean": [14.2, 15.1, 13.7],
            "temperature_2m_max": [18.0, 19.2, 17.6],
            "temperature_2m_min": [10.1, 11.0, 9.9],
            "apparent_temperature_mean": [13.5, 14.6, 12.8],
            "wind_speed_10m_max": [22.0, 18.0, 35.0],
            "sunshine_duration": [18000, 20000, 6000],
            "precipitation_sum": [0.0, 0.2, 8.3],
            "shortwave_radiation_sum": [9.1, 10.4, 3.0],
        },
        "timezone": "Europe/Paris",
    }

@pytest.fixture
def multi_year_df():
    """DataFrame tối thiểu cho forecast/PCA: ~2 năm ngày-ngày."""
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    n = len(dates)
    # Tạo dữ liệu synthetic có tính mùa vụ nhẹ
    t = np.arange(n)
    temp = 12 + 8*np.sin(2*np.pi*t/365.0) + np.random.RandomState(0).normal(0, 0.5, size=n)
    df = pd.DataFrame({
        "date": dates,
        "temperature_2m_mean": temp,
        "apparent_temperature_mean": temp - 0.8,
        "wind_speed_10m_max": 20 + 5*np.sin(2*np.pi*t/20.0),
        "sunshine_duration": np.clip(15000 + 5000*np.sin(2*np.pi*t/365.0), 0, None),
        "precipitation_sum": np.clip(np.random.RandomState(1).gamma(1.5, 1.0, size=n)-0.5, 0, None),
        "shortwave_radiation_sum": np.clip(8 + 4*np.sin(2*np.pi*t/365.0), 0, None),
    })
    return df

# Fakes cho WeatherService unit test
class _FakeGeocoder:
    def __init__(self, geoloc=None):
        self.geoloc = geoloc or {"latitude": 45.76, "longitude": 4.84, "timezone": "Europe/Paris"}
    def geocode(self, city: str):
        return self.geoloc

class _FakeWeatherProvider:
    def __init__(self, today_json, range_json, last_year_json):
        self._today = today_json
        self._range = range_json
        self._last_year = last_year_json
    def daily_today(self, geoloc): return self._today
    def daily_range(self, geoloc, start, end): return self._range
    def daily_same_day_last_year(self, geoloc, date_last_year): return self._last_year

@pytest.fixture
def fake_geocoder():
    return _FakeGeocoder()

@pytest.fixture
def fake_provider(sample_daily_json):
    # Dùng cùng một mẫu JSON cho cả 3 đường dẫn để đơn giản
    return _FakeWeatherProvider(
        today_json=sample_daily_json,
        range_json=sample_daily_json,
        last_year_json=sample_daily_json,
    )
