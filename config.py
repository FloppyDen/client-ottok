# -*- coding: utf-8 -*-
"""
Файл конфигурации проекта.
Содержит все константы, пути и настройки для воспроизводимости результатов.
"""

import os

# ==================== ПУТИ И ДИРЕКТОРИИ ====================
# URL датасета Telco Customer Churn (открытый источник IBM)
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

# Локальный путь для сохранения датасета (на случай офлайн-работы)
DATA_PATH = "data/telco_customer_churn.csv"

# Директория для сохранения результатов (графики, модели)
RESULTS_DIR = "results"

# Создаём директории, если их нет
os.makedirs("data", exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==================== ПАРАМЕТРЫ РАЗДЕЛЕНИЯ ДАННЫХ ====================
# Размер тестовой выборки (20% данных)
TEST_SIZE = 0.2

# Фиксированное случайное состояние для воспроизводимости
RANDOM_STATE = 42

# Количество фолдов для кросс-валидации
CV_FOLDS = 5

# ==================== ПАРАМЕТРЫ МОДЕЛЕЙ ====================
# Максимальное количество итераций для линейных моделей
MAX_ITER = 1000

# Параметры сетки для GridSearch (CatBoost)
CATBOOST_PARAM_GRID = {
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'iterations': [100, 200]
}

# ==================== НАСТРОЙКИ ВИЗУАЛИЗАЦИИ ====================
# Размер фигур по умолчанию
FIGSIZE_DEFAULT = (10, 6)
FIGSIZE_LARGE = (14, 8)

# Стиль графиков seaborn
SEABORN_STYLE = "whitegrid"

# Цветовая палитра
PALETTE = "viridis"