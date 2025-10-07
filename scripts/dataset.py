"""
Создание объединенного датасета из свечей и новостей
Объединяет данные по дате и тикеру для обучения моделей прогнозирования
"""

import pandas as pd
import numpy as np
import ast
import random
from datetime import datetime, timedelta
import os
from config import (
    TRAIN_CANDLES_PATH,
    OUTPUT_FILE_PATH,
    DATA_DIR,
    COMBINED_DATASET_PATH,
    SEED
)

# Фиксируем seed для воспроизводимости
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

def parse_affected_tickers(tickers_str):
    """
    Парсит строку с тикерами из формата ['TICKER1', 'TICKER2'] в список
    """
    if pd.isna(tickers_str) or tickers_str == '[]':
        return []
    
    try:
        if isinstance(tickers_str, str):
            return ast.literal_eval(tickers_str)
        return tickers_str
    except (ValueError, SyntaxError):
        return []

def prepare_news_data(news_df):
    """
    Подготавливает данные новостей для объединения
    """
    
    # Конвертируем дату публикации в datetime
    news_df['publish_date'] = pd.to_datetime(news_df['publish_date'])
    
    news_df['publish_date_only'] = news_df['publish_date'].dt.date
    
    news_df['affected_tickers_parsed'] = news_df['affected_tickers'].apply(parse_affected_tickers)
    
    # Создаем отдельные строки для каждого тикера в affected_tickers
    expanded_news = []
    
    for idx, row in news_df.iterrows():
        tickers = row['affected_tickers_parsed']
        if tickers:  # Если есть тикеры
            for ticker in tickers:
                expanded_news.append({
                    'publish_date': row['publish_date'],
                    'publish_date_only': row['publish_date_only'],
                    'ticker': ticker,
                    'sentiment': row['sentiment'],
                    'importance': row['importance'],
                    'category': row['category'],
                    'original_index': row['original_index']
                })
        else:
            expanded_news.append({
                'publish_date': row['publish_date'],
                'publish_date_only': row['publish_date_only'],
                'ticker': None,
                'sentiment': row['sentiment'],
                'importance': row['importance'],
                'category': row['category'],
                'original_index': row['original_index']
            })
    
    expanded_news_df = pd.DataFrame(expanded_news)
    
    return expanded_news_df

def prepare_candles_data(candles_df):
    """
    Подготавливает данные свечей для объединения
    """
    candles_df['begin'] = pd.to_datetime(candles_df['begin'])
    candles_df['begin_date_only'] = candles_df['begin'].dt.date
    
    return candles_df

def get_all_categories(news_df):
    """
    Получает все уникальные категории из новостей для one-hot encoding
    """
    all_categories = set()
    for categories_list in news_df['category']:
        if isinstance(categories_list, str):
            try:
                categories = ast.literal_eval(categories_list)
                if isinstance(categories, list):
                    all_categories.update(categories)
            except:
                all_categories.add(categories_list)
        elif isinstance(categories_list, list):
            all_categories.update(categories_list)
    
    return sorted(list(all_categories))

def aggregate_news_features(news_df):
    """
    Агрегирует признаки новостей по дате и тикеру
    """
    
    all_categories = get_all_categories(news_df)
    
    # Группируем по дате и тикеру
    grouped = news_df.groupby(['publish_date_only', 'ticker']).agg({
        'sentiment': ['count', 'std'],
        'importance': ['max', 'count'],
        'category': lambda x: list(x),
        'original_index': 'count'
    }).reset_index()
    
    # Упрощаем названия колонок
    grouped.columns = [
        'date', 'ticker',
        'news_count', 'news_sentiment_std',
        'news_importance_max', 'news_importance_count',
        'news_categories', 'news_original_count'
    ]
    
    # Заполняем NaN значения
    grouped['news_sentiment_std'] = grouped['news_sentiment_std'].fillna(0)
    
    # Создаем one-hot encoding для категорий
    for category in all_categories:
        grouped[f'news_category_{category.replace("/", "_").replace(" ", "_")}'] = 0
    
    # Заполняем one-hot признаки
    for idx, row in grouped.iterrows():
        categories_list = row['news_categories']
        if isinstance(categories_list, list):
            for category in categories_list:
                if isinstance(category, str):
                    try:
                        parsed_categories = ast.literal_eval(category)
                        if isinstance(parsed_categories, list):
                            for cat in parsed_categories:
                                col_name = f'news_category_{cat.replace("/", "_").replace(" ", "_")}'
                                if col_name in grouped.columns:
                                    grouped.at[idx, col_name] = 1
                        else:
                            col_name = f'news_category_{category.replace("/", "_").replace(" ", "_")}'
                            if col_name in grouped.columns:
                                grouped.at[idx, col_name] = 1
                    except:
                        col_name = f'news_category_{category.replace("/", "_").replace(" ", "_")}'
                        if col_name in grouped.columns:
                            grouped.at[idx, col_name] = 1
                elif isinstance(category, list):
                    for cat in category:
                        col_name = f'news_category_{cat.replace("/", "_").replace(" ", "_")}'
                        if col_name in grouped.columns:
                            grouped.at[idx, col_name] = 1
    
    # Создаем дополнительные признаки
    grouped['has_high_importance_news'] = (grouped['news_importance_max'] >= 8).astype(int)
    grouped['has_any_news'] = (grouped['news_count'] > 0).astype(int)
    
    # Удаляем исходную колонку с категориями
    grouped = grouped.drop('news_categories', axis=1)
    
    return grouped

def merge_candles_with_news(candles_df, news_aggregated_df):
    """
    Объединяет данные свечей с агрегированными новостями
    Учитывает задержку новостей на 1 день (новости пишутся через день после изменений)
    """
    
    # Сдвигаем даты новостей на 1 день назад (новости пишутся через день после изменений)
    news_aggregated_df['date_shifted'] = pd.to_datetime(news_aggregated_df['date']) - timedelta(days=1)
    news_aggregated_df['date_shifted'] = news_aggregated_df['date_shifted'].dt.date
    
    # Объединяем по сдвинутой дате и тикеру
    merged_df = candles_df.merge(
        news_aggregated_df,
        left_on=['begin_date_only', 'ticker'],
        right_on=['date_shifted', 'ticker'],
        how='left'
    )
    
    # Удаляем служебные колонки
    if 'date' in merged_df.columns:
        merged_df = merged_df.drop('date', axis=1)
    if 'date_shifted' in merged_df.columns:
        merged_df = merged_df.drop('date_shifted', axis=1)
    
    # Заполняем NaN значения для дней без новостей
    news_columns = [col for col in merged_df.columns if col.startswith('news_')]
    for col in news_columns:
        if col in ['news_sentiment_std', 'news_importance_max']:
            merged_df[col] = merged_df[col].fillna(0)
        elif col in ['news_count', 'news_importance_count', 'news_original_count']:
            merged_df[col] = merged_df[col].fillna(0)
        elif col.startswith('news_category_'):
            merged_df[col] = merged_df[col].fillna(0)
        elif col in ['has_high_importance_news', 'has_any_news']:
            merged_df[col] = merged_df[col].fillna(0)
    
    return merged_df

def create_combined_dataset():
    """
    Создает объединенный датасет из свечей и новостей
    """
    
    candles_df = pd.read_csv(TRAIN_CANDLES_PATH)
    news_df = pd.read_csv(OUTPUT_FILE_PATH)
    
    candles_prepared = prepare_candles_data(candles_df)
    news_prepared = prepare_news_data(news_df)
    
    news_aggregated = aggregate_news_features(news_prepared)
    
    combined_df = merge_candles_with_news(candles_prepared, news_aggregated)
    
    combined_df.to_csv(COMBINED_DATASET_PATH, index=False)
    
    return combined_df

if __name__ == "__main__":
    combined_dataset = create_combined_dataset()
