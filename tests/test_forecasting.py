# tests/test_forecasting.py
import pandas as pd
from services.analytics.forecasting import forecast_temperature_next_year

def test_forecast_next_year_length_and_start(multi_year_df):
    out = forecast_temperature_next_year(multi_year_df, periods=365)
    assert list(out.columns) == ["date", "temperature_2m_mean_predite"]
    assert len(out) == 365
    # ngày đầu tiên = ngày cuối cùng trong input + 1
    last = pd.to_datetime(multi_year_df["date"]).max()
    assert pd.to_datetime(out["date"].iloc[0]) == last + pd.Timedelta(days=1)
