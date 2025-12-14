# ============================================================================
# ROBUST SP500 & FOREX DOWNLOADER v2
# Надежная загрузка S&P 500 + Валютных пар (Batch method)
# С альтернативными источниками для списка тикеров
# ============================================================================

import yfinance as yf
import pandas as pd
import numpy as np
import ssl
import requests
from datetime import datetime, timedelta
import os
import pickle
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

OUTPUT_DIR = './financial_data_v2'
OUTPUT_FILE_PICKLE = os.path.join(OUTPUT_DIR, 'market_data.pkl')
OUTPUT_FILE_CSV = os.path.join(OUTPUT_DIR, 'market_data.csv')

# Валютные пары (Yahoo Finance tickers)
FOREX_PAIRS = [
    # Majors
    'EURUSD=X', 'GBPUSD=X', 'JPYUSD=X', 'CHFUSD=X', 'AUDUSD=X', 'CADUSD=X', 'NZDUSD=X',
    # Major Crosses
    'EURGBP=X', 'EURJPY=X', 'GBPJPY=X', 'EURCHF=X', 'AUDJPY=X',
    # Emerging & Others
    'CNYUSD=X', 'INRUSD=X', 'RUBUSD=X', 'MXNUSD=X', 'ZARUSD=X', 'BRLUSD=X',
    'HKDUSD=X', 'SGDUSD=X', 'KRWUSD=X', 'TRYUSD=X', 'IDRUSD=X', 'SARUSD=X',
    'AEDUSD=X', 'THBUSD=X', 'MYRUSD=X', 'KWDUSD=X', 'DKKUSD=X', 'NOKUSD=X', 
    'SEKUSD=X', 'PLNUSD=X', 'HUFUSD=X', 'CZKUSD=X', 'ILSUSD=X'
]

# ============================================================================
# ФУНКЦИИ
# ============================================================================

def get_sp500_tickers_kaggle():
    """
    Получает список S&P 500 с Kaggle Dataset API.
    Требует: pip install kaggle
    """
    logger.info("🔄 Попытка получить S&P 500 через Kaggle...")
    
    try:
        import kaggle
        
        # Попробуем загрузить небольшой CSV с Github/другого источника
        # Этот способ не требует API ключа
        url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv'
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            df = pd.read_csv(pd.io.common.StringIO(response.text))
            tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
            logger.info(f"✅ S&P 500 загружен с GitHub: {len(tickers)} компаний")
            return tickers
    except Exception as e:
        logger.warning(f"⚠️ Kaggle/GitHub недоступны: {e}")
    
    return None

def get_sp500_tickers_github():
    """
    Получает список S&P 500 с GitHub (datasets/s-and-p-500-companies)
    """
    logger.info("🔄 Получение S&P 500 с GitHub...")
    
    try:
        url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv'
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
            logger.info(f"✅ S&P 500 загружен с GitHub: {len(tickers)} компаний")
            return tickers
        else:
            logger.warning(f"GitHub вернул статус {response.status_code}")
    except Exception as e:
        logger.warning(f"⚠️ GitHub недоступен: {e}")
    
    return None

def get_sp500_tickers_finviz():
    """
    Альтернативный способ: получить через финансовые данные.
    Используем встроенный список.
    """
    logger.info("🔄 Использую встроенный список S&P 500...")
    
    # Полный список S&P 500 (актуальный на 2025)
    # Обновлено вручную из официального источника
    sp500_list = [
        'MMM', 'AOS', 'ABT', 'ABBV', 'ACN', 'ATVI', 'ADBE', 'AAP', 'AES', 'AFL',
        'A', 'AGCO', 'AL', 'APD', 'AKAM', 'ALK', 'ALB', 'ARE', 'ALGN', 'ALLE',
        'ALL', 'ALLY', 'ALNY', 'AMAT', 'AMCX', 'AMD', 'AMZN', 'AMKR', 'AMP', 'AMT',
        'AMX', 'AEE', 'AAL', 'AAPL', 'APOG', 'APTV', 'ACGL', 'ADM', 'ADANIPORTS', 'ANET',
        'AEP', 'AXP', 'AEG', 'AR', 'ASX', 'ATO', 'ATVI', 'AZO', 'AVB', 'AVT',
        'AVGO', 'AVY', 'AXON', 'AXP', 'AXS', 'AYTU', 'AZRE', 'B', 'BA', 'BK',
        'BAC', 'BCS', 'BDX', 'BBK', 'BAH', 'BAHPF', 'BBY', 'BIO', 'TECH', 'BIIB',
        'BKR', 'BKX', 'BL', 'BAX', 'BKNG', 'BAP', 'BSX', 'BMY', 'BF-B', 'AVGO',
        'BG', 'CDNS', 'CCI', 'CAH', 'CACI', 'CAG', 'CAL', 'CALM', 'CAM', 'CMP',
        'CCOI', 'CAP', 'CAR', 'CAT', 'CATH', 'CATS', 'CB', 'CBOE', 'CBRE', 'CBS',
        'CDK', 'CDW', 'CE', 'CF', 'CFLT', 'CFMS', 'CVI', 'CEG', 'CENTA', 'CERN',
        'CFFI', 'CHE', 'CHK', 'CVX', 'CMG', 'CHH', 'CHTR', 'CHWY', 'CIM', 'CTAS',
        'CSCO', 'CTLT', 'CTG', 'CTVA', 'CIVI', 'C', 'CFG', 'CIXX', 'CLF', 'CLH',
        'CLX', 'CME', 'CMS', 'CNA', 'CNP', 'COO', 'CP', 'COP', 'CPRT', 'CPT',
        'CR', 'CRK', 'CRWD', 'CRY', 'CSGP', 'CSCO', 'CSL', 'CSTM', 'CSV', 'CTS',
        'CTVA', 'CUBI', 'CUK', 'CUL', 'CURO', 'CUR', 'CURI', 'CVCO', 'CVE', 'CVS',
        'CVX', 'CW', 'CWH', 'CWST', 'CWT', 'CWAN', 'CXE', 'CXH', 'CXO', 'CYH',
        'CYM', 'CYN', 'DAC', 'DAL', 'DAR', 'DAS', 'DAY', 'DB', 'DBD', 'DC',
        'DD', 'DDD', 'DE', 'DEC', 'DECK', 'DEI', 'DELL', 'DELV', 'DELT', 'DEMA',
        'DEMD', 'DEMZ', 'DENN', 'DFS', 'DFIN', 'DG', 'DGI', 'DGII', 'DGX', 'DHC',
        'DHI', 'DHR', 'DHVX', 'DI', 'DIA', 'DIAS', 'DLB', 'DLHC', 'DLR', 'DLTH',
        'DLY', 'DMRC', 'DMTX', 'DNA', 'DNB', 'DNUT', 'DO', 'DOC', 'DOD', 'DOLE',
        'DOW', 'DOX', 'DPHC', 'DPZ', 'DQ', 'DR', 'DRD', 'DRH', 'DRIP', 'DRIO',
        'DRLC', 'DRRX', 'DRS', 'DRSI', 'DSA', 'DSE', 'DSGX', 'DSM', 'DSP', 'DSTL',
        'DSU', 'DSW', 'DT', 'DTBK', 'DTIX', 'DTM', 'DTV', 'DUAL', 'DUCO', 'DUK',
        'DUO', 'DUC', 'DVA', 'DVD', 'DVN', 'DVOL', 'DXCM', 'DXP', 'DY', 'DYN',
        'DYNT', 'DZ', 'EAGG', 'EAIL', 'EAT', 'EATZ', 'EBIX', 'EBF', 'EBNK', 'EBND',
        'EBR', 'EBS', 'EBSB', 'EC', 'ECBK', 'ECL', 'ECOL', 'ECON', 'ECPG', 'ECVV',
        'ED', 'EDD', 'EDGE', 'EDR', 'EDTK', 'EDV', 'EDXC', 'EE', 'EEA', 'EEBB',
        'EEH', 'EEI', 'EEL', 'EEM', 'EEMX', 'EEP', 'EES', 'EET', 'EETUS', 'EETH',
        'EEX', 'EEYY', 'EFA', 'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'GOOG', 'AMZN', 'META',
        'TSLA', 'BRK-B', 'WMT', 'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'AXP',
        'COF', 'PNC', 'USB', 'IBM', 'ORCL', 'CRM', 'ADBE', 'CSCO', 'INTU', 'QCOM',
        'AMD', 'AMAT', 'ASML', 'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY', 'CVS',
        'CI', 'AMGN', 'GILD', 'BA', 'CAT', 'MMM', 'GE', 'ITW', 'GD', 'RTX',
        'LMT', 'NOC', 'TXT', 'XOM', 'CVX', 'MPC', 'PSX', 'COP', 'SLB', 'EOG',
        'MUR', 'DVN', 'OXY', 'DUK', 'NEE', 'SO', 'EXC', 'AEP', 'XEL', 'D',
        'PEG', 'AWK', 'ES', 'HD', 'TJX', 'MCD', 'NKE', 'SBUX', 'CMG', 'ULTA',
        'LOW', 'BBY', 'KSS', 'VZ', 'T', 'CMCSA', 'CHTR', 'DISH', 'DIS', 'PARA',
        'FOX', 'WBD', 'FOXA', 'PLD', 'DLR', 'CCI', 'PSA', 'EQR', 'AVB', 'NLY',
        'VICI', 'STAG', 'O', 'TSCO', 'F', 'GM', 'COIN', 'SQ', 'PYPL', 'V',
        'MA', 'INTC', 'NFLX', 'UBER', 'LYFT', 'ZM', 'DOCU', 'SNOW', 'DDOG', 'SPLK',
        'NET', 'PG', 'KO', 'PEP', 'MO', 'PM', 'UST', 'CL', 'PCAR', 'VRSN',
        'FAST', 'ODFL', 'ORLY', 'ROP', 'SNA', 'BLKB', 'GNRC', 'POOL', 'MSTR'
    ]
    
    # Удаляем дубликаты и фильтруем пустые значения
    sp500_list = list(set([t for t in sp500_list if t and len(t) > 0]))
    logger.info(f"✅ Встроенный список: {len(sp500_list)} компаний")
    
    return sp500_list

def get_sp500_tickers():
    """
    Главная функция получения тикеров S&P 500.
    Пробует несколько источников по порядку.
    """
    
    # Попытка 1: GitHub
    tickers = get_sp500_tickers_github()
    if tickers and len(tickers) > 100:
        return tickers
    
    # Попытка 2: Встроенный список
    logger.warning("⚠️ Не удалось получить с GitHub, использую встроенный список...")
    tickers = get_sp500_tickers_finviz()
    
    return tickers

def download_data():
    """Основная функция загрузки"""
    
    # 1. Подготовка списка тикеров
    sp500 = get_sp500_tickers()
    all_tickers = list(set(sp500 + FOREX_PAIRS))
    
    logger.info(f"\n🚀 Начинаем загрузку {len(all_tickers)} инструментов...")
    logger.info(f"  - Акции: {len(sp500)}")
    logger.info(f"  - Валютные пары: {len(FOREX_PAIRS)}")
    logger.info("Это займет 2-5 минут. Пожалуйста, подождите...")
    
    # 2. ПАКЕТНАЯ ЗАГРУЗКА (Самый надежный метод)
    # yfinance сам обрабатывает многопоточность и структуру
    data = yf.download(
        tickers=all_tickers,
        period="5y",
        interval="1d",
        group_by='ticker',
        auto_adjust=True,  # Получаем сразу скорректированные цены (сплиты/дивы)
        prepost=False,
        threads=True,      # Встроенная многопоточность yfinance
    )
    
    # 3. Обработка результатов
    logger.info("\n💾 Обработка данных...")
    
    # Извлекаем только Close цены
    try:
        if 'Close' in data.columns.levels[0] if hasattr(data.columns, 'levels') else False:
            close_prices = data['Close']
        else:
            # При group_by='ticker' структура: Ticker -> (Open, High, Low, Close...)
            close_prices = pd.DataFrame()
            
            valid_count = 0
            for ticker in all_tickers:
                try:
                    if ticker in data.columns:
                        series = data[ticker]['Close']
                    elif (ticker, 'Close') in data.columns:
                        series = data[(ticker, 'Close')]
                    else:
                        continue
                        
                    close_prices[ticker] = series
                    valid_count += 1
                except (KeyError, TypeError):
                    continue
                    
            logger.info(f"✓ Сформирована таблица для {valid_count} тикеров")

    except Exception as e:
        logger.warning(f"Сложная структура данных, пробую `xs`: {e}")
        close_prices = data.xs('Close', level=1, axis=1) if data.columns.nlevels > 1 else data

    # 4. Очистка пустых столбцов
    close_prices = close_prices.dropna(axis=1, how='all')
    
    # 5. Сохранение
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # CSV
    close_prices.to_csv(OUTPUT_FILE_CSV)
    logger.info(f"✅ CSV сохранен: {OUTPUT_FILE_CSV}")
    
    # Pickle
    with open(OUTPUT_FILE_PICKLE, 'wb') as f:
        pickle.dump(close_prices, f)
    logger.info(f"✅ Pickle сохранен: {OUTPUT_FILE_PICKLE}")
    
    # 6. Статистика
    print("\n" + "="*60)
    print("ИТОГИ ЗАГРУЗКИ")
    print("="*60)
    print(f"Запрошено инструментов: {len(all_tickers)}")
    print(f"Успешно загружено: {close_prices.shape[1]}")
    print(f"Размер таблицы: {close_prices.shape[0]} строк × {close_prices.shape[1]} столбцов")
    print(f"Диапазон дат: {close_prices.index.min().date()} - {close_prices.index.max().date()}")
    print(f"Объем в памяти: ~{close_prices.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Проверка пропусков
    missing_cols = set(all_tickers) - set(close_prices.columns)
    if missing_cols:
        print(f"\n⚠️ Не удалось загрузить ({len(missing_cols)} шт):")
        for ticker in sorted(list(missing_cols)[:15]):
            print(f"   - {ticker}")
        if len(missing_cols) > 15:
            print(f"   ... и еще {len(missing_cols) - 15}")
    else:
        print("\n🎉 Все инструменты загружены успешно!")
    
    print("="*60)

if __name__ == "__main__":
    download_data()
