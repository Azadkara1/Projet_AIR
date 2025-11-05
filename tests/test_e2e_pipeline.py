import pandas as pd
import numpy as np
import pytest

from data.transformer import DataTransformer
from services.analytics.forecasting import forecast_temperature_next_year
from services.analytics.pca import acp_temperature
from services.weather_service import WeatherService

class DummyGeocoder:
    def geocode(self, city: str):
        return {"latitude": 45.76, "longitude": 4.84, "timezone": "Europe/Paris"}

class DummyProvider:
    def _mk_range(self, start, end):
        dates = pd.date_range(start, end, freq="D")
        n = len(dates); t = np.arange(n)
        temp = 12 + 8*np.sin(2*np.pi*t/365.0)
        return {
            "daily": {
                "time": dates.strftime("%Y-%m-%d").tolist(),
                "temperature_2m_mean": temp.tolist(),
                "apparent_temperature_mean": (temp-0.7).tolist(),
                "wind_speed_10m_max": (20 + 5*np.sin(2*np.pi*t/20.0)).tolist(),
                "sunshine_duration": (15000 + 4000*np.sin(2*np.pi*t/365.0)).tolist(),
                "precipitation_sum": (np.clip(np.random.RandomState(1).gamma(1.5, 1.0, size=n)-0.5, 0, None)).tolist(),
                "shortwave_radiation_sum": (8 + 3*np.sin(2*np.pi*t/365.0)).tolist(),
            },
            "timezone": "Europe/Paris",
        }

    def daily_today(self, geoloc):
        return self._mk_range("2024-10-03", "2024-10-03")

    def daily_same_day_last_year(self, geoloc, date_last_year):
        return self._mk_range(date_last_year, date_last_year)

    def daily_range(self, geoloc, start, end):
        return self._mk_range(start, end)

@pytest.mark.order("last")
def test_e2e_pipeline_ok():
    svc = WeatherService(geocoder=DummyGeocoder(), provider=DummyProvider(), transformer=DataTransformer())

    df = svc.get_multi_year_data("Lyon", years=2, end_date="2024-10-03")
    assert isinstance(df, pd.DataFrame) and not df.empty

    fc = forecast_temperature_next_year(df, periods=30)  # 30 ngày cho nhanh
    assert len(fc) == 30

    pcs, loadings, explained = acp_temperature(df, "2024-06-01", "2024-08-31")
    assert pcs.shape[0] > 0 and loadings.shape[0] >= 4 and explained.size >= 2
