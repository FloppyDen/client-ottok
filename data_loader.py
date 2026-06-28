# -*- coding: utf-8 -*-
"""
Модуль загрузки и предобработки данных.
Выполняет пункты 1-3 задания курсовой работы:
  - загрузка датасета;
  - первичный анализ;
  - очистка и преобразование признаков.
"""

import os
import pandas as pd
import numpy as np
from config import DATA_URL, DATA_PATH


def load_data() -> pd.DataFrame:
    """
    Загружает датасет Telco Customer Churn.
    Сначала пытается загрузить из интернета, при неудаче — из локального файла.
    
    Returns:
        pd.DataFrame: исходный набор данных.
    """
    print("=" * 60)
    print("ШАГ 1. ЗАГРУЗКА ДАННЫХ")
    print("=" * 60)
    
    try:
        # Пытаемся загрузить с GitHub
        df = pd.read_csv(DATA_URL)
        print(f"[OK] Данные успешно загружены из интернета.")
        print(f"[OK] Сохраняем локальную копию в {DATA_PATH}...")
        df.to_csv(DATA_PATH, index=False)
    except Exception as e:
        # Если интернет недоступен — читаем локальный файл
        print(f"[!] Не удалось загрузить из интернета: {e}")
        print(f"[OK] Загружаем локальную копию: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    
    print(f"[i] Размер датасета: {df.shape[0]} строк, {df.shape[1]} колонок.")
    return df


def display_dataset_info(df: pd.DataFrame) -> None:
    """
    Выводит общую информацию о датасете: типы данных, пропуски, примеры.
    """
    print("\n--- Общая информация о датасете ---")
    df.info()
    
    print("\n--- Статистическое описание числовых признаков ---")
    print(df.describe())
    
    print("\n--- Первые 5 строк датасета ---")
    print(df.head())
    
    print("\n--- Наличие пропущенных значений ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Пропусков': missing,
        'Процент': missing_pct
    })
    print(missing_df[missing_df['Пропусков'] > 0])


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очищает датасет:
      - обрабатывает пропуски в TotalCharges;
      - преобразует типы данных;
      - удаляет нерелевантные колонки (customerID).
    
    Args:
        df: исходный DataFrame.
    
    Returns:
        pd.DataFrame: очищенный датасет.
    """
    print("\n" + "=" * 60)
    print("ШАГ 2. ОЧИСТКА ДАННЫХ")
    print("=" * 60)
    
    # Удаляем идентификатор клиента — он не несёт информации для модели
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        print("[OK] Колонка 'customerID' удалена.")
    
    # TotalCharges содержит пробелы вместо NaN — преобразуем в числовой тип
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Пропуски в TotalCharges возникают у клиентов с tenure=0 (новый клиент)
    # Заменяем их на 0, так как дохода ещё не было
    missing_tc = df['TotalCharges'].isnull().sum()
    df['TotalCharges'].fillna(0.0, inplace=True)
    print(f"[OK] Заполнено {missing_tc} пропусков в TotalCharges нулями.")
    
    print(f"[OK] Очистка завершена. Итоговый размер: {df.shape}")
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Преобразует целевую переменную Churn из Yes/No в 1/0.
    """
    print("\n--- Кодирование целевой переменной ---")
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Проверяем баланс классов
    churn_dist = df['Churn'].value_counts(normalize=True) * 100
    print(f"[i] Распределение классов:")
    print(f"    Не ушли (0): {churn_dist[0]:.2f}%")
    print(f"    Ушли (1):    {churn_dist[1]:.2f}%")
    print(f"[!] Присутствует дисбаланс классов (~26% оттока).")
    return df


def prepare_features(df: pd.DataFrame):
    """
    Разделяет данные на признаки (X) и целевую переменную (y).
    Применяет One-Hot Encoding для категориальных признаков.
    
    Returns:
        tuple: (X, y, feature_names)
    """
    print("\n" + "=" * 60)
    print("ШАГ 3. ПОДГОТОВКА ПРИЗНАКОВ")
    print("=" * 60)
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # One-Hot Encoding с drop_first для избежания мультиколлинеарности
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    print(f"[OK] После One-Hot Encoding: {X_encoded.shape[1]} признаков.")
    print(f"[i] Названия признаков: {list(X_encoded.columns)}")
    
    return X_encoded, y, list(X_encoded.columns)