# -*- coding: utf-8 -*-
"""
Модуль оптимизации гиперпараметров.
Выполняет пункт 9 задания: GridSearch, анализ важности признаков.
"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, recall_score
from catboost import CatBoostClassifier
from config import RESULTS_DIR, CATBOOST_PARAM_GRID, CV_FOLDS, RANDOM_STATE, FIGSIZE_DEFAULT


def optimize_catboost(X_train, y_train, X_test, y_test, feature_names: list):
    """
    Оптимизирует гиперпараметры CatBoost с помощью GridSearchCV.
    
    Returns:
        tuple: (best_model, grid_results_dict)
    """
    print("\n" + "=" * 60)
    print("ШАГ 7. ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ (CatBoost)")
    print("=" * 60)
    
    print(f"[i] Сетка параметров: {CATBOOST_PARAM_GRID}")
    
    base_model = CatBoostClassifier(
        verbose=0,
        random_state=RANDOM_STATE
    )
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=CATBOOST_PARAM_GRID,
        cv=CV_FOLDS,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    print("\n[i] Запуск GridSearchCV...")
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    print(f"\n[OK] GridSearch завершён за {search_time:.2f} сек.")
    print(f"[OK] Лучшие параметры: {grid_search.best_params_}")
    print(f"[OK] Лучший ROC-AUC (CV): {grid_search.best_score_:.4f}")
    
    best_model = grid_search.best_estimator_
    
    # Оценка на тестовой выборке
    y_pred_opt = best_model.predict(X_test)
    y_prob_opt = best_model.predict_proba(X_test)[:, 1]
    
    test_roc_auc = roc_auc_score(y_test, y_prob_opt)
    test_recall = recall_score(y_test, y_pred_opt)
    
    print(f"\n[i] Результаты оптимизированной модели на тесте:")
    print(f"    ROC-AUC: {test_roc_auc:.4f}")
    print(f"    Recall:  {test_recall:.4f}")
    
    # Визуализация результатов GridSearch
    plot_grid_search_results(grid_search)
    
    # Анализ важности признаков
    plot_feature_importance(best_model, feature_names)
    
    return best_model, {
        'best_params': grid_search.best_params_,
        'best_cv_score': grid_search.best_score_,
        'test_roc_auc': test_roc_auc,
        'test_recall': test_recall,
        'search_time': search_time
    }


def plot_grid_search_results(grid_search: GridSearchCV) -> None:
    """Визуализирует результаты поиска по сетке параметров."""
    sns.set_style("whitegrid")
    results = pd.DataFrame(grid_search.cv_results_)
    
    # Группируем по глубине дерева и строим график
    plt.figure(figsize=FIGSIZE_DEFAULT)
    
    for depth in results['param_depth'].unique():
        subset = results[results['param_depth'] == depth]
        plt.plot(subset['param_learning_rate'], 
                 subset['mean_test_score'],
                 marker='o', label=f'depth={depth}')
    
    plt.xlabel('Learning Rate')
    plt.ylabel('Mean ROC-AUC (CV)')
    plt.title('Результаты GridSearch: влияние learning_rate и depth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/10_grid_search_results.png", bbox_inches='tight')
    plt.show()
    print(f"[OK] График сохранён: {RESULTS_DIR}/10_grid_search_results.png")


def plot_feature_importance(model, feature_names: list, top_n: int = 15) -> None:
    """
    Строит график важности признаков для CatBoost.
    """
    sns.set_style("whitegrid")
    
    importances = model.get_feature_importance()
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df, 
                palette='viridis')
    plt.title(f'Топ-{top_n} наиболее важных признаков (CatBoost)')
    plt.xlabel('Важность')
    plt.ylabel('Признак')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/11_feature_importance.png", bbox_inches='tight')
    plt.show()
    print(f"[OK] График сохранён: {RESULTS_DIR}/11_feature_importance.png")
    
    print("\n[i] Топ-5 признаков по важности:")
    for i, row in importance_df.head(5).iterrows():
        print(f"    {i+1}. {row['Feature']}: {row['Importance']:.2f}")