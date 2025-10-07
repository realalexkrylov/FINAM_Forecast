import pandas as pd
import numpy as np
import os
import random
import joblib
from datetime import timedelta
from config import TARGET_DAYS, LAGS, WINDOWS, TICKER_COL, SAVE_DIR, DATA_DIR, COMBINED_DATASET_PATH, SUBMISSION_PATH, SEED

# Фиксируем seed для воспроизводимости
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

def prepare_features_for_future(df, lag_days=LAGS, windows=WINDOWS):
    """Подготавливает признаки для будущих предсказаний"""
    df = df.sort_values(["ticker", "begin"]).reset_index(drop=True)
    all_features = []

    for ticker, group in df.groupby("ticker"):
        g = group.copy()
        
        # Технические индикаторы
        for lag in lag_days:
            g[f"close_lag_{lag}"] = g["close"].shift(lag)
            g[f"volume_lag_{lag}"] = g["volume"].shift(lag)
        
        for window in windows:
            g[f"close_ma_{window}"] = g["close"].rolling(window).mean()
            g[f"close_std_{window}"] = g["close"].rolling(window).std()
            g[f"volume_ma_{window}"] = g["volume"].rolling(window).mean()
            g[f"volume_std_{window}"] = g["volume"].rolling(window).std()
        
        g["close_diff_1"] = g["close"].diff(1)
        g["close_diff_5"] = g["close"].diff(5)
        
        # Дополнительные технические признаки
        g["high_low_ratio"] = g["high"] / g["low"]
        g["open_close_ratio"] = g["open"] / g["close"]
        g["volume_price_ratio"] = g["volume"] / g["close"]
        
        # Заполняем NaN значения
        g = g.ffill().bfill()
        all_features.append(g)

    return pd.concat(all_features, axis=0).reset_index(drop=True)

def generate_future_dates(candles, target_days=TARGET_DAYS):
    """Генерирует календарные дни для предсказаний с новостными признаками"""
    future_data_list = []
    last_dates = candles.groupby("ticker")["begin"].max().to_dict()
    tickers = candles[TICKER_COL].unique()

    for ticker in tickers:
        last_date = last_dates[ticker]
        # Генерируем календарные дни (включая выходные)
        dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=target_days*2, freq='D')
        
        df = pd.DataFrame({
            "begin": dates,
            "ticker": ticker,
            "close": np.nan,
            "volume": np.nan,
            "open": np.nan,
            "high": np.nan,
            "low": np.nan
        })
        
        # Добавляем новостные признаки
        news_columns = [col for col in candles.columns if col.startswith('news_') or col.startswith('has_')]
        for col in news_columns:
            df[col] = 0.0
        
        future_data_list.append(df)

    return pd.concat(future_data_list, axis=0).reset_index(drop=True)

# Функция предсказаний
def make_predictions(future_prepared, candles, save_dir=SAVE_DIR, target_days=TARGET_DAYS):
    """Делает предсказания для всех тикеров"""
    returns_dict = {}
    tickers = candles[TICKER_COL].unique()

    for ticker in tickers:
        
        model_path = os.path.join(save_dir, f"{ticker}_model.pkl")
        features_path = os.path.join(save_dir, f"{ticker}_features.pkl")
        
        if not os.path.exists(model_path):
            continue
            
        model = joblib.load(model_path)
        feature_info = joblib.load(features_path)
        feature_cols = feature_info['feature_cols']

        df_ticker = future_prepared[future_prepared["ticker"] == ticker].copy()
        
        missing_features = [col for col in feature_cols if col not in df_ticker.columns]
        if missing_features:
            for col in missing_features:
                df_ticker[col] = 0.0

        df_ticker = df_ticker.head(target_days)
        
        # Заполняем NaN значения в признаках
        df_ticker[feature_cols] = df_ticker[feature_cols].fillna(0)
        
        y_pred = model.predict(df_ticker[feature_cols])
        df_ticker["pred_close"] = y_pred

        close_series = df_ticker["pred_close"].values
        dates_series = df_ticker["begin"].values
        
        # Получаем последнюю цену для данного тикера
        last_real_close = candles[candles["ticker"] == ticker]["close"].iloc[-1]
        
        returns = [np.nan] * target_days
        
        # Заполняем доходности только для будних дней
        for i in range(target_days):
            current_date = dates_series[i]
            weekday = pd.Timestamp(current_date).weekday()
            if weekday < 5:
                returns[i] = close_series[i] / last_real_close - 1
        
        returns_dict[ticker] = returns
    
    return returns_dict

def make_submission(returns_dict, target_days=TARGET_DAYS, output_path=SUBMISSION_PATH):
    """Создает матрицу доходностей и сохраняет в CSV"""
    returns_matrix = pd.DataFrame(
        returns_dict,
        index=[f"p{i}" for i in range(1, target_days+1)]
    ).T
    returns_matrix.index.name = "ticker"

    returns_matrix.to_csv(output_path, na_rep='NaN')


def main():
    """Основная функция для создания submission файла"""
    candles = pd.read_csv(COMBINED_DATASET_PATH, parse_dates=["begin"])
    
    # Генерация будущих дат
    future_data = generate_future_dates(candles)
    
    full_data = pd.concat([candles, future_data], axis=0).reset_index(drop=True)
    
    # Подготавливаем признаки
    full_data_prepared = prepare_features_for_future(full_data)
    
    last_date = candles["begin"].max()
    future_prepared = full_data_prepared[full_data_prepared["begin"] > last_date]
    
    returns_dict = make_predictions(future_prepared, candles)
    
    make_submission(returns_dict)

if __name__ == "__main__":
    main()