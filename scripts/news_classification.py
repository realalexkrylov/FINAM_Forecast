import os
import json
import time
import asyncio
import aiohttp
import pandas as pd
import requests
import numpy as np
import random
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_URL,
    MODEL,
    CONCURRENT_REQUESTS,
    MAX_RETRIES,
    RETRY_DELAY,
    TRAIN_CANDLES_PATH,
    TRAIN_NEWS_PATH,
    OUTPUT_FILE_PATH,
    SEED
)

# Фиксируем seed для воспроизводимости
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

def get_company_name(ticker):
    """Получение названия компании по тикеру с MOEX"""
    url = f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{ticker}.json"
    response = requests.get(url)
    data = response.json()
    securities = data['securities']['data']
    columns = data['securities']['columns']
    secname_index = columns.index('SECNAME')
    company_names = [sec[secname_index] for sec in securities]
    return company_names[0]

async def extract_news_features_async(session, title, publication, publish_date, semaphore, idx, tickers, helper):
    """
    Асинхронное извлечение признаков из новости используя LLM через OpenRouter API
    
    Параметры:
    - session: aiohttp.ClientSession
    - title: заголовок новости
    - publication: текст новости
    - publish_date: дата публикации новости
    - semaphore: asyncio.Semaphore для контроля параллелизма
    - idx: индекс новости
    - tickers: список тикеров
    - helper: словарь соответствия тикеров и названий компаний
    
    Возвращает словарь с признаками
    """
    
    # Ограничиваем длину текста для экономии токенов
    text_sample = publication[:1500] if len(publication) > 1500 else publication
    
    prompt = f"""Проанализируй следующую финансовую новость и верни JSON с признаками:

Заголовок: {title}
Текст: {text_sample}

Проанализируй новость и верни JSON в формате:

{{
    "sentiment": <число от -1 (очень негативная новость) до 1 (очень позитивная)>,
    "importance": <целое число от 0 до 10, насколько новость важна для рынка>,
    "category": <одна или несколько категорий из списка: "макроэкономика", "компания", "сектор", "регуляция", "геополитика", "дивиденды", "финансы", "прочее", "отчётность/доходы", "слияния/поглощения/корпоративные события", "менеджмент/кадровые изменения", "продукт/контракты/инновации", "судебные/регуляторные/санкции", "конкуренция/рыночная доля", "события/форс-мажор", "рейтинги/аналитика">,
    "affected_tickers": <список тикеров компаний из [{', '.join(tickers)}] которых касается новость, или [] если компания не упомянута>
}}

Правила:
- sentiment отражает общий настрой (падение/санкции = ближе к -1, рост/прибыль/новые проекты = ближе к 1, нейтральные макроновости = около 0).
- importance = оценивай от 0 (неважная, мало влияющая) до 10 (крупное событие, санкции, изменения в налогах, банкротства, большие дивиденды).
- category может содержать несколько меток, если новость одновременно относится к разным аспектам (например, "отчётность/доходы" и "дивиденды").
- affected_tickers = укажи тикеры компаний, если они явно названы или однозначно связаны с новостью.

Справочник тикеров и компаний (используй для сопоставления):
{';  '.join([f'{ticker} = {value}' for ticker, value in helper.items()])}
"""

    async with semaphore:  # Ограничиваем количество параллельных запросов
        for attempt in range(MAX_RETRIES):
            try:
                async with session.post(
                    OPENROUTER_URL,
                    json={
                        "model": MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0,
                        "max_tokens": 500
                    },
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"HTTP {response.status}: {error_text}")
                    result_data = await response.json()
                    result_text = result_data['choices'][0]['message']['content'].strip()
                    
                    # Попытка извлечь JSON из ответа
                    if "```json" in result_text:
                        result_text = result_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in result_text:
                        result_text = result_text.split("```")[1].split("```")[0].strip()
                    
                    result = json.loads(result_text)
                    
                    # Валидация результата
                    required_keys = ["sentiment", "importance", "category", "affected_tickers"]
                    if all(key in result for key in required_keys):
                        result['original_index'] = idx
                        result['publish_date'] = publish_date  # Добавляем дату сразу
                        return result
                    else:
                        raise ValueError(f"Отсутствуют ключи: {set(required_keys) - set(result.keys())}")
                        
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    return {
                        "sentiment": 0.0,
                        "importance": 5,
                        "category": "прочее",
                        "affected_tickers": [],
                        "original_index": idx,
                        "publish_date": publish_date
                    }
    
    return None

async def process_news_batch_async(df, tickers, helper, max_news=None):
    """
    Асинхронная обработка новостей с параллельными запросами
    
    Параметры:
    - df: датафрейм с новостями
    - tickers: список тикеров
    - helper: словарь соответствия тикеров и названий компаний
    - max_news: максимальное количество новостей для обработки (None = все)
    
    Возвращает датафрейм с новыми признаками
    """
    
    if max_news:
        df = df.head(max_news)
    
    # Создаем semaphore для ограничения параллелизма
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    
    async with aiohttp.ClientSession() as session:
        # Создаем задачи для всех новостей
        tasks = []
        for idx, row in df.iterrows():
            title = row.get('title', '')
            publication = row.get('publication', '')
            publish_date = row.get('publish_date', '')  # Получаем дату из исходного датафрейма
            task = extract_news_features_async(session, title, publication, publish_date, semaphore, idx, tickers, helper)
            tasks.append(task)
        
        results = []
        for i, coro in enumerate(async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="⚡ Обработка")):
            result = await coro
            results.append(result)
    
    features_df = pd.DataFrame(results)
    
    return features_df

def load_data():
    """Загрузка данных"""
    
    train_candles = pd.read_csv(TRAIN_CANDLES_PATH)
    tickers = train_candles['ticker'].unique().tolist()
    
    helper = {ticker: get_company_name(ticker) for ticker in tickers}
    
    train_news = pd.read_csv(TRAIN_NEWS_PATH)
    
    return train_candles, train_news, tickers, helper

async def main():
    """Основная функция"""
    train_candles, train_news, tickers, helper = load_data()
    
    # Запускаем асинхронную обработку
    news_features = await process_news_batch_async(
        train_news, 
        tickers,
        helper,
        max_news=None
    )
    
    news_features.to_csv(OUTPUT_FILE_PATH, index=False)
    
    return news_features

if __name__ == "__main__":
    news_features = asyncio.run(main())

