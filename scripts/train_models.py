import pandas as pd
import os
import numpy as np
import random
import joblib
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import uniform, randint
from config import TARGET_DAYS, LAGS, WINDOWS, TICKER_COL, SAVE_DIR, DATA_DIR, COMBINED_DATASET_PATH, SEED

# Фиксируем seed для воспроизводимости
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

def prepare_data(df, lags=LAGS, windows=WINDOWS):
    """Подготавливает данные с техническими индикаторами и новостными признаками"""
    df = df.sort_values([TICKER_COL, "begin"]).reset_index(drop=True)
    all_features = []

    for ticker, group in df.groupby(TICKER_COL):
        g = group.copy()
        
        # Технические индикаторы
        for lag in lags:
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
        
        # Заполняем NaN пропущенные значения
        g = g.ffill().bfill()
        all_features.append(g)
    
    return pd.concat(all_features, axis=0).reset_index(drop=True)

def check_gpu_support():
    """Проверяет доступность GPU для LightGBM"""
    try:
        import lightgbm as lgb
        test_model = lgb.LGBMRegressor(
            device='gpu', 
            gpu_platform_id=0, 
            gpu_device_id=0, 
            n_estimators=1,
            random_state=SEED
        )
        # Пробуем обучить модель на тестовых данных
        import numpy as np
        X_test = np.random.rand(10, 5)
        y_test = np.random.rand(10)
        test_model.fit(X_test, y_test)
        return True
    except Exception as e:
        print(f"GPU не поддерживается: {e}")
        return False

def train_models(train_data, tickers, save_dir=SAVE_DIR):
    """Обучает модели для каждого тикера с быстрым перебором параметров"""
    models = {}
    
    # Проверяем поддержку GPU
    use_gpu = check_gpu_support()
    print(f"GPU поддержка: {'Да' if use_gpu else 'Нет'}")
    
    # Параметры для случайного поиска
    param_distributions = {
        'n_estimators': randint(400, 1000),
        'learning_rate': uniform(0.01, 0.2),
        'num_leaves': randint(20, 100),
        'max_depth': randint(4, 12),
        'min_child_samples': randint(10, 50),
        'subsample': uniform(0.7, 0.3),
        'colsample_bytree': uniform(0.7, 0.3),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0, 1)
    }
    
    for ticker in tickers:
        print(f"Обучаем модель для {ticker}...")
        data = train_data[train_data[TICKER_COL] == ticker].copy()
        
        exclude_cols = ["begin", "ticker", "close", "begin_date_only", "open", "high", "low", "volume"]
        feature_cols = [c for c in data.columns if c not in exclude_cols]
        
        X_train = data[feature_cols]
        y_train = data["close"]
        
        # Базовые настройки модели LightGBM с GPU
        base_params = {
            'random_state': SEED,
            'verbose': -1,
            'n_jobs': -1
        }
        
        if use_gpu:
            base_params.update({
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0
            })
        
        base_model = LGBMRegressor(**base_params)
        
        # Используем TimeSeriesSplit для временных рядов
        tscv = TimeSeriesSplit(n_splits=3)

        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=5,  # Количество случайных комбинаций
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=0,
            random_state=SEED
        )
        
        random_search.fit(X_train, y_train)
        
        # Получаем лучшую модель
        best_model = random_search.best_estimator_
        models[ticker] = best_model
        
        model_path = os.path.join(save_dir, f"{ticker}_model.pkl")
        joblib.dump(best_model, model_path)
        
        # Сохраняем список признаков для использования в prediction
        feature_info = {
            'feature_cols': feature_cols,
            'ticker': ticker,
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_
        }
        joblib.dump(feature_info, os.path.join(save_dir, f"{ticker}_features.pkl"))
    
    return models

if __name__ == "__main__":
    combined_data = pd.read_csv(COMBINED_DATASET_PATH, parse_dates=["begin"])
    
    # Подготавливаем данные с признаками
    train_data = prepare_data(combined_data)
    
    # Получаем список тикеров
    tickers = combined_data[TICKER_COL].unique()
    
    models = train_models(train_data, tickers)
