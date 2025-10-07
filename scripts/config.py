import os

TARGET_DAYS = 20 # количество дней вперед, на которое мы прогнозируем
LAGS = [1, 2, 3, 5, 10] # лаги
WINDOWS = [3, 5, 10] # скользящие окна
TICKER_COL = "ticker"
SAVE_DIR = "./ticker_models"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

TRAIN_CANDLES_PATH = os.path.join(DATA_DIR, "candles.csv")
TRAIN_NEWS_PATH = os.path.join(DATA_DIR, "news.csv")
OUTPUT_FILE_PATH = os.path.join(DATA_DIR, "train_news_features.csv")
COMBINED_DATASET_PATH = os.path.join(DATA_DIR, "combined_dataset.csv")

DATA_PATH = "../data/candles.csv"
NEWS_PATH = "../data/news.csv"
NEWS_FEATURES_PATH = "../data/news_features.csv"
SUBMISSION_PATH = "../data/submission.csv"

# OpenRouter API настройки
OPENROUTER_API_KEY = "token"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-4o-mini"  # Оптимальный баланс скорости и точности

CONCURRENT_REQUESTS = 100  # Количество параллельных запросов
MAX_RETRIES = 2
RETRY_DELAY = 1
SEED = 42
