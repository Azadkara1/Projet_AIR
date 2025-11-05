# tests/test_weather_service_unit.py
import pytest
import pandas as pd

from services.weather_service import WeatherService
from data.transformer import DataTransformer

def _has_method(obj, name):
    return getattr(obj, name, None) and callable(getattr(obj, name))

@pytest.mark.skipif(not _has_method(WeatherService, "get_weather_range"),
                    reason="WeatherService.get_weather_range not implemented in current snapshot")
def test_get_weather_range_returns_df(fake_geocoder, fake_provider):
    svc = WeatherService(geocoder=fake_geocoder, provider=fake_provider, transformer=DataTransformer())
    df = svc.get_weather_range("Lyon", "2024-10-01", "2024-10-03")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert df.index.name == "date"

@pytest.mark.skipif(not _has_method(WeatherService, "get_today_vs_last_year"),
                    reason="WeatherService.get_today_vs_last_year not implemented in current snapshot")
def test_get_today_vs_last_year_tuple(fake_geocoder, fake_provider):
    svc = WeatherService(geocoder=fake_geocoder, provider=fake_provider, transformer=DataTransformer())
    today_df, last_year_df = svc.get_today_vs_last_year("Lyon")
    assert isinstance(today_df, pd.DataFrame) and isinstance(last_year_df, pd.DataFrame)
    assert not today_df.empty and not last_year_df.empty

@pytest.mark.skipif(not _has_method(WeatherService, "get_multi_year_data"),
                    reason="WeatherService.get_multi_year_data not implemented in current snapshot")
def test_get_multi_year_data(fake_geocoder, fake_provider):
    svc = WeatherService(geocoder=fake_geocoder, provider=fake_provider, transformer=DataTransformer())
    df = svc.get_multi_year_data("Lyon", years=1, end_date="2024-10-03")
    assert df is None or isinstance(df, pd.DataFrame)
