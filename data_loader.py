# -*- coding: utf-8 -*-
"""
Модуль загрузки и предобработки данных.
"""

import os
import pandas as pd
import numpy as np
from config import DATA_URL, DATA_PATH


def load_data() -> pd.DataFrame:
    """Загружает датасет Telco Customer Churn."""
    print("=" * 60)
    print("ШАГ 1. ЗАГРУЗКА ДАННЫХ")
    print("=" * 60)
    
    try:
        df = pd.read_csv(DATA_URL)
        print(f"[OK] Данные успешно загружены из интернета.")
        print(f"[OK] Сохраняем локальную копию в {DATA_PATH}...")
        df.to_csv(DATA_PATH, index=False)
    except Exception as e:
        print(f"[!] Не удалось загрузить из интернета: {e}")
        print(f"[OK] Загружаем локальную копию: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    
    print(f"[i] Размер датасета: {df.shape[0]} строк, {df.shape[1]} колонок.")
    return df


def display_dataset_info(df: pd.DataFrame) -> None:
    """Выводит общую информацию о датасете."""
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
    Очищает датасет. Гарантированно убирает все NaN.
    """
    print("\n" + "=" * 60)
    print("ШАГ 2. ОЧИСТКА ДАННЫХ")
    print("=" * 60)
    
    # Удаляем идентификатор клиента
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        print("[OK] Колонка 'customerID' удалена.")
    
    # Преобразуем TotalCharges в числовой тип
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Заполняем пропуски в TotalCharges нулями (явное присваивание!)
    missing_tc = df['TotalCharges'].isnull().sum()
    df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
    print(f"[OK] Заполнено {missing_tc} пропусков в TotalCharges нулями.")
    
    # КРИТИЧЕСКИ ВАЖНО: заполняем ВСЕ остальные пропуски во всём DataFrame
    total_missing = df.isnull().sum().sum()
    if total_missing > 0:
        print(f"[!] Обнаружено ещё {total_missing} пропусков в данных. Заполняем...")
        # Для числовых — медиана
        for col in df.select_dtypes(include=['number']).columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        # Для категориальных — мода
        for col in df.select_dtypes(exclude=['number']).columns:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].fillna(mode_val)
    
    # Финальная проверка
    final_missing = df.isnull().sum().sum()
    print(f"[OK] Очистка завершена. Итоговый размер: {df.shape}")
    print(f"[OK] Осталось пропусков: {final_missing}")
    
    if final_missing > 0:
        print("[!!!] ВНИМАНИЕ: В данных всё ещё есть пропуски!")
        print(df.isnull().sum()[df.isnull().sum() > 0])
    
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Преобразует целевую переменную Churn из Yes/No в 1/0."""
    print("\n--- Кодирование целевой переменной ---")
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    churn_dist = df['Churn'].value_counts(normalize=True) * 100
    print(f"[i] Распределение классов:")
    print(f"    Не ушли (0): {churn_dist[0]:.2f}%")
    print(f"    Ушли (1):    {churn_dist[1]:.2f}%")
    print(f"[!] Присутствует дисбаланс классов (~26% оттока).")
    return df


def prepare_features(df: pd.DataFrame):
    """
    Разделяет данные на признаки (X) и целевую переменную (y).
    Применяет One-Hot Encoding. Гарантированно убирает все NaN.
    """
    print("\n" + "=" * 60)
    print("ШАГ 3. ПОДГОТОВКА ПРИЗНАКОВ")
    print("=" * 60)
    
    # ПРОВЕРКА: есть ли NaN перед кодированием?
    if df.isnull().sum().sum() > 0:
        print(f"[!] Обнаружено {df.isnull().sum().sum()} пропусков перед кодированием!")
        print(df.isnull().sum()[df.isnull().sum() > 0])
        # Заполняем числовые медианой, категориальные модой
        for col in df.select_dtypes(include=['number']).columns:
            df[col] = df[col].fillna(df[col].median())
        for col in df.select_dtypes(exclude=['number']).columns:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].fillna(mode_val)
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # One-Hot Encoding
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # КРИТИЧЕСКИ ВАЖНО: после get_dummies проверяем на NaN
    if X_encoded.isnull().sum().sum() > 0:
        print(f"[!] После get_dummies осталось {X_encoded.isnull().sum().sum()} пропусков. Заполняем нулями.")
        X_encoded = X_encoded.fillna(0)
    
    print(f"[OK] После One-Hot Encoding: {X_encoded.shape[1]} признаков.")
    print(f"[OK] Пропусков в X: {X_encoded.isnull().sum().sum()}")
    
    return X_encoded, y, list(X_encoded.columns)