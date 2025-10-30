import pandas as pd

def create_weather_dataframe(api_response, include_all_fields=False):
    if not api_response or "hourly" not in api_response:
        return pd.DataFrame()

    hourly = api_response["hourly"]

    if include_all_fields:
        df = pd.DataFrame(hourly)
    else:
        columns_needed = [
            "time",
            "temperature_2m",
            "pressure_msl",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_gusts_10m",
            "precipitation",
            "apparent_temperature",
            "cloud_cover",
            "wind_direction_10m"
        ]
        df = pd.DataFrame({col: hourly[col] for col in columns_needed if col in hourly})
    
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
    
    return df

def create_daily_dataframe(api_response):
    if not api_response or "daily" not in api_response:
        return pd.DataFrame()
    
    daily = api_response["daily"]
    df = pd.DataFrame(daily)
    
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
    
    return df

def merge_hourly_and_daily(hourly_df, daily_df):
    if hourly_df.empty:
        return hourly_df
    if daily_df.empty:
        return hourly_df
    
    hourly_with_date = hourly_df.copy()
    hourly_with_date['date'] = hourly_with_date.index.date
    
    daily_renamed = daily_df.copy()
    daily_renamed.index = daily_renamed.index.date
    daily_renamed = daily_renamed.add_prefix('daily_')
    
    merged = hourly_with_date.merge(
        daily_renamed,
        left_on='date',
        right_index=True,
        how='left'
    )
    
    merged.drop('date', axis=1, inplace=True)
    
    return merged