# -*- coding: utf-8 -*-
"""
Модуль визуализации результатов сравнения моделей.
Выполняет пункт 10 задания.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from config import RESULTS_DIR, FIGSIZE_DEFAULT, FIGSIZE_LARGE, SEABORN_STYLE


def setup_style():
    """Устанавливает стиль seaborn."""
    sns.set_style(SEABORN_STYLE)


def plot_metrics_comparison(results_df: pd.DataFrame) -> None:
    """
    Строит сравнительные графики метрик для всех моделей.
    """
    setup_style()
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        df_sorted = results_df.sort_values(by=metric, ascending=True)
        colors = ['#3498db' if m not in ['XGBoost', 'LightGBM', 'CatBoost', 'VotingClassifier']
                  else '#e74c3c' for m in df_sorted['Model']]
        
        df_sorted.plot(kind='barh', x='Model', y=metric, ax=axes[i], 
                       color=colors, legend=False)
        axes[i].set_title(f'Сравнение по {metric}')
        axes[i].set_xlabel(metric)
        axes[i].set_xlim(0, 1.05)
        axes[i].axvline(x=0.8, color='gray', linestyle='--', alpha=0.5)
    
    # График времени обучения
    df_sorted_time = results_df.sort_values(by='Train Time (s)', ascending=True)
    df_sorted_time.plot(kind='barh', x='Model', y='Train Time (s)', 
                        ax=axes[5], color='#9b59b6', legend=False)
    axes[5].set_title('Время обучения (сек)')
    axes[5].set_xlabel('Время (сек)')
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/06_metrics_comparison.png", bbox_inches='tight')
    plt.show()
    print(f"[OK] График сохранён: {RESULTS_DIR}/06_metrics_comparison.png")


def plot_roc_auc_comparison(models_dict: dict, X_test, y_test) -> None:
    """
    Строит ROC-кривые для всех моделей на одном графике.
    """
    setup_style()
    
    plt.figure(figsize=(10, 8))
    
    for name, model in models_dict.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Случайный классификатор')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривые всех моделей')
    plt.legend(loc='lower right', fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/07_roc_curves.png", bbox_inches='tight')
    plt.show()
    print(f"[OK] График сохранён: {RESULTS_DIR}/07_roc_curves.png")


def plot_confusion_matrices(models_dict: dict, X_test, y_test) -> None:
    """
    Строит матрицы ошибок для ключевых моделей.
    """
    setup_style()
    key_models = ['RandomForest', 'XGBoost', 'CatBoost', 'VotingClassifier']
    available = [m for m in key_models if m in models_dict]
    
    n = len(available)
    if n == 0:
        return
    
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, name in enumerate(available):
        model = models_dict[name]
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Остался', 'Ушёл'],
                    yticklabels=['Остался', 'Ушёл'])
        axes[i].set_title(f'Матрица ошибок: {name}')
        axes[i].set_xlabel('Предсказано')
        axes[i].set_ylabel('Фактически')
    
    # Скрываем пустые подграфики
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/08_confusion_matrices.png", bbox_inches='tight')
    plt.show()
    print(f"[OK] График сохранён: {RESULTS_DIR}/08_confusion_matrices.png")


def plot_learning_curves(model, X_train, y_train, model_name: str) -> None:
    """
    Строит кривые обучения для оценки переобучения.
    """
    setup_style()
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        cv=5, scoring='roc_auc',
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    plt.figure(figsize=FIGSIZE_DEFAULT)
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='#3498db')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                     alpha=0.1, color='#e74c3c')
    plt.plot(train_sizes, train_mean, 'o-', color='#3498db', label='Обучающая выборка')
    plt.plot(train_sizes, val_mean, 'o-', color='#e74c3c', label='Валидационная выборка')
    
    plt.xlabel('Размер обучающей выборки')
    plt.ylabel('ROC-AUC')
    plt.title(f'Кривые обучения: {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/09_learning_curve_{model_name}.png", bbox_inches='tight')
    plt.show()
    print(f"[OK] График сохранён: {RESULTS_DIR}/09_learning_curve_{model_name}.png")