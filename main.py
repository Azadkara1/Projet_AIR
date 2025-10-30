from datetime import datetime, timedelta
from DataCollecte.api_client import get_geocoding_data, get_hourly_weather_data, get_historical_weather_data
from DataCollecte.data_cleaning import create_weather_dataframe, create_daily_dataframe


def main():
    city = "lyon"
    
    today = datetime.now()
    one_year_ago = today - timedelta(days=365)
    
    today_str = today.strftime("%Y-%m-%d")
    one_year_ago_str = one_year_ago.strftime("%Y-%m-%d")
    
    geolocalisation = get_geocoding_data(city)
    
    response_today = get_hourly_weather_data(geolocalisation, today_str, today_str)
    df_today_hourly = create_weather_dataframe(response_today)
    df_today_daily = create_daily_dataframe(response_today)
    
    response_last_year = get_historical_weather_data(geolocalisation, one_year_ago_str, one_year_ago_str)
    df_last_year_hourly = create_weather_dataframe(response_last_year)
    df_last_year_daily = create_daily_dataframe(response_last_year)
    
    print(df_today_hourly)
    print(df_today_daily)
    print(df_last_year_hourly)
    print(df_last_year_daily)

    return {
        'today': {
            'hourly': df_today_hourly,
            'daily': df_today_daily
        },
        'last_year': {
            'hourly': df_last_year_hourly,
            'daily': df_last_year_daily
        }
    }



if __name__ == "__main__":
    data = main()

