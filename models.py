# -*- coding: utf-8 -*-
"""
Модуль создания, обучения и оценки моделей.
Выполняет пункты 5-8 задания:
  - создание 7+ базовых моделей;
  - создание ансамблевых моделей;
  - обучение и предсказание;
  - замер времени обучения.
"""

import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

from config import TEST_SIZE, RANDOM_STATE, CV_FOLDS, MAX_ITER


def split_data(X, y):
    """
    Разделяет данные на обучающую и тестовую выборки.
    Используется стратифицированное разбиение для сохранения баланса классов.
    """
    print("\n" + "=" * 60)
    print("ШАГ 5. РАЗДЕЛЕНИЕ НА ВЫБОРКИ")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    print(f"[OK] Обучающая выборка: {X_train.shape}")
    print(f"[OK] Тестовая выборка:  {X_test.shape}")
    print(f"[i] Метод разделения: стратифицированное случайное (80/20).")
    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test):
    """
    Стандартизирует числовые признаки (важно для SVM, KNN, LogReg).
    """
    print("\n--- Стандартизация признаков ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("[OK] Применён StandardScaler.")
    return X_train_scaled, X_test_scaled, scaler


def get_base_models() -> dict:
    """
    Создаёт словарь из 7 базовых моделей (требование задания).
    """
    return {
        'LogisticRegression': LogisticRegression(
            max_iter=MAX_ITER, random_state=RANDOM_STATE, solver='lbfgs'
        ),
        'SVC': SVC(
            probability=True, random_state=RANDOM_STATE, kernel='rbf'
        ),
        'KNeighbors': KNeighborsClassifier(n_neighbors=5),
        'GaussianNB': GaussianNB(),
        'DecisionTree': DecisionTreeClassifier(
            random_state=RANDOM_STATE, max_depth=10
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, max_depth=10
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100, random_state=RANDOM_STATE, max_depth=5
        )
    }


def get_ensemble_models(base_models: dict) -> dict:
    """
    Создаёт ансамблевые модели: XGBoost, LightGBM, CatBoost, Voting.
    """
    return {
        'XGBoost': XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=RANDOM_STATE,
            eval_metric='logloss',
            use_label_encoder=False
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=RANDOM_STATE,
            verbose=-1
        ),
        'CatBoost': CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=6,
            random_state=RANDOM_STATE,
            verbose=0
        ),
        'VotingClassifier': VotingClassifier(
            estimators=[
                ('rf', base_models['RandomForest']),
                ('gbm', base_models['GradientBoosting']),
                ('lr', base_models['LogisticRegression'])
            ],
            voting='soft'
        )
    }


def train_and_evaluate(models_dict: dict, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """
    Обучает все модели из словаря, замеряет время, вычисляет метрики.
    Используется кросс-валидация для оценки стабильности.
    
    Returns:
        pd.DataFrame: таблица с результатами по каждой модели.
    """
    results = []
    
    print("\n" + "=" * 60)
    print("ШАГ 6. ОБУЧЕНИЕ МОДЕЛЕЙ И ЗАМЕР ВРЕМЕНИ")
    print("=" * 60)
    
    for name, model in models_dict.items():
        print(f"\n[i] Обучение модели: {name}...")
        start_time = time.time()
        
        # Кросс-валидация (ROC-AUC) для оценки стабильности
        try:
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=CV_FOLDS, scoring='roc_auc', n_jobs=-1
            )
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        except Exception as e:
            print(f"[!] Ошибка кросс-валидации для {name}: {e}")
            cv_mean, cv_std = np.nan, np.nan
        
        # Обучение на полном трейне
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Предсказание на тесте
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Вычисление метрик
        results.append({
            'Model': name,
            'Accuracy': round(accuracy_score(y_test, y_pred), 4),
            'Precision': round(precision_score(y_test, y_pred), 4),
            'Recall': round(recall_score(y_test, y_pred), 4),
            'F1': round(f1_score(y_test, y_pred), 4),
            'ROC-AUC': round(roc_auc_score(y_test, y_prob), 4),
            'CV ROC-AUC (mean)': round(cv_mean, 4),
            'CV ROC-AUC (std)': round(cv_std, 4),
            'Train Time (s)': round(train_time, 3)
        })
        
        print(f"    [OK] ROC-AUC: {results[-1]['ROC-AUC']:.4f} | "
              f"Время: {train_time:.2f} сек")
    
    return pd.DataFrame(results)


def print_classification_report(model, X_test, y_test, model_name: str) -> None:
    """Выводит подробный отчёт по классификации и матрицу ошибок."""
    print(f"\n--- Подробный отчёт для {model_name} ---")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Остался', 'Ушёл']))
    
    cm = confusion_matrix(y_test, y_pred)
    print("Матрица ошибок:")
    print(f"              Предсказано: Остался  Ушёл")
    print(f"  Фактически Остался:  {cm[0][0]:5d}   {cm[0][1]:5d}")
    print(f"             Ушёл:     {cm[1][0]:5d}   {cm[1][1]:5d}")