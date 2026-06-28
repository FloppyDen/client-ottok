# -*- coding: utf-8 -*-
"""
Модуль описательного анализа данных (EDA).
Выполняет пункт 4 задания: визуализация распределений, корреляций, 
совместных распределений с целевой переменной.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import RESULTS_DIR, FIGSIZE_DEFAULT, FIGSIZE_LARGE, SEABORN_STYLE, PALETTE


def setup_plot_style():
    """Устанавливает единый стиль для всех графиков."""
    sns.set_style(SEABORN_STYLE)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['figure.dpi'] = 100


def plot_target_distribution(y: pd.Series) -> None:
    """Визуализирует распределение целевой переменной."""
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DEFAULT)
    
    # Счётчик
    y.value_counts().plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
    axes[0].set_title('Распределение классов (абсолютные значения)')
    axes[0].set_xlabel('Churn')
    axes[0].set_ylabel('Количество клиентов')
    axes[0].set_xticklabels(['Не ушёл (0)', 'Ушёл (1)'], rotation=0)
    
    # Процентное соотношение
    y.value_counts(normalize=True).plot(kind='pie', ax=axes[1], 
                                         autopct='%1.1f%%',
                                         colors=['#2ecc71', '#e74c3c'])
    axes[1].set_title('Доля классов')
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/01_target_distribution.png", bbox_inches='tight')
    plt.show()
    print(f"[OK] График сохранён: {RESULTS_DIR}/01_target_distribution.png")


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    """Строит гистограммы числовых признаков."""
    setup_plot_style()
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_LARGE)
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], bins=30, kde=True, ax=axes[i], color='#3498db')
        axes[i].set_title(f'Распределение: {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Частота')
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/02_numeric_distributions.png", bbox_inches='tight')
    plt.show()
    print(f"[OK] График сохранён: {RESULTS_DIR}/02_numeric_distributions.png")


def plot_churn_by_category(df: pd.DataFrame) -> None:
    """Показывает долю оттока в разрезе категориальных признаков."""
    setup_plot_style()
    categorical_cols = ['Contract', 'InternetService', 'PaymentMethod', 'OnlineSecurity']
    
    # Оставляем только те колонки, которые есть в датасете
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    
    n = len(categorical_cols)
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_LARGE)
    axes = axes.flatten()
    
    for i, col in enumerate(categorical_cols[:4]):
        churn_rate = df.groupby(col)['Churn'].mean() * 100
        churn_rate.plot(kind='bar', ax=axes[i], color='#e67e22')
        axes[i].set_title(f'Доля оттока по {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('% оттока')
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
        axes[i].axhline(y=df['Churn'].mean() * 100, color='red', 
                        linestyle='--', label='Средний отток')
        axes[i].legend()
    
    # Скрываем пустые подграфики, если их меньше 4
    for j in range(len(categorical_cols), 4):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/03_churn_by_category.png", bbox_inches='tight')
    plt.show()
    print(f"[OK] График сохранён: {RESULTS_DIR}/03_churn_by_category.png")


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """Строит матрицу корреляций числовых признаков."""
    setup_plot_style()
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']
    corr = df[numeric_cols].corr()
    
    plt.figure(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', 
                center=0, fmt='.2f', square=True)
    plt.title('Матрица корреляций числовых признаков')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/04_correlation_matrix.png", bbox_inches='tight')
    plt.show()
    print(f"[OK] График сохранён: {RESULTS_DIR}/04_correlation_matrix.png")


def plot_tenure_vs_churn(df: pd.DataFrame) -> None:
    """Анализирует зависимость оттока от срока обслуживания."""
    setup_plot_style()
    plt.figure(figsize=FIGSIZE_DEFAULT)
    
    sns.kdeplot(data=df[df['Churn'] == 0]['tenure'], 
                label='Оставшиеся клиенты', fill=True, color='#2ecc71')
    sns.kdeplot(data=df[df['Churn'] == 1]['tenure'], 
                label='Ушедшие клиенты', fill=True, color='#e74c3c')
    
    plt.title('Распределение срока обслуживания (tenure) по классам')
    plt.xlabel('Срок обслуживания (месяцы)')
    plt.ylabel('Плотность')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/05_tenure_vs_churn.png", bbox_inches='tight')
    plt.show()
    print(f"[OK] График сохранён: {RESULTS_DIR}/05_tenure_vs_churn.png")


def run_full_eda(df: pd.DataFrame, y: pd.Series) -> None:
    """
    Запускает полный цикл описательного анализа.
    
    Args:
        df: исходный DataFrame (до кодирования).
        y: целевая переменная.
    """
    print("\n" + "=" * 60)
    print("ШАГ 4. ОПИСАТЕЛЬНЫЙ АНАЛИЗ ДАННЫХ (EDA)")
    print("=" * 60)
    
    # Восстанавливаем исходный Churn для визуализации (если уже закодирован)
    df_viz = df.copy()
    if df_viz['Churn'].dtype in ['int64', 'int32', 'float64']:
        df_viz['Churn'] = df_viz['Churn'].map({1: 'Yes', 0: 'No'})
    
    plot_target_distribution(y)
    plot_numeric_distributions(df)
    plot_churn_by_category(df)
    plot_correlation_matrix(df)
    plot_tenure_vs_churn(df)
    
    print("\n[✓] Описательный анализ завершён.")
    print("[i] Ключевые выводы:")
    print("    1. Дисбаланс классов: ~26% клиентов уходят.")
    print("    2. Короткий срок обслуживания (tenure) коррелирует с оттоком.")
    print("    3. Высокие ежемесячные платежи увеличивают вероятность оттока.")
    print("    4. Клиенты с ежемесячным контрактом уходят чаще.")