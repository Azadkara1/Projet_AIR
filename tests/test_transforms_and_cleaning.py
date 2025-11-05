# tests/test_transforms_and_cleaning.py
import pandas as pd
import data.transforms
import data.data_cleaning
from data.transformer import DataTransformer

def test_create_daily_dataframe_via_module(sample_daily_json):
    df = data.transforms.create_daily_dataframe(sample_daily_json)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "temperature_2m_mean" in df.columns
    # đã set index là 'date'
    assert df.index.name == "date"
    assert pd.Timestamp("2024-10-01") in df.index

def test_create_daily_dataframe_missing_key():
    # Không có "daily" -> trả DataFrame rỗng (đặc tả hiện có)
    assert data.transforms.create_daily_dataframe({}).empty

def test_data_cleaning_module_compat(sample_daily_json):
    # bản cũ còn file data_cleaning.py -> vẫn hoạt động tương tự
    df = data.data_cleaning.create_daily_dataframe(sample_daily_json)
    assert not df.empty

def test_transformer_class(sample_daily_json):
    tr = DataTransformer()
    df = tr.create_daily_dataframe(sample_daily_json)
    assert {"weathercode","precipitation_sum"}.issubset(df.columns)
