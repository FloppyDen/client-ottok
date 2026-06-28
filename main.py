# -*- coding: utf-8 -*-
"""
Главный исполняемый файл проекта.
Запускает полный цикл курсовой работы:
  от загрузки данных до оптимизации модели.
  
Автор: Студент
Тема: Применение ансамблевых методов для анализа клиентского оттока
Дисциплина: Машинное обучение и анализ данных
"""

import warnings
import pandas as pd
from config import RESULTS_DIR

# Подавляем предупреждения для чистоты вывода
warnings.filterwarnings('ignore')

# Импорт собственных модулей
from data_loader import (
    load_data, display_dataset_info, 
    clean_data, encode_target, prepare_features
)
from eda import run_full_eda
from models import (
    split_data, scale_data,
    get_base_models, get_ensemble_models,
    train_and_evaluate, print_classification_report
)
from evaluation import (
    plot_metrics_comparison, plot_roc_auc_comparison,
    plot_confusion_matrices, plot_learning_curves
)
from optimization import optimize_catboost


def main():
    """Основная функция, выполняющая все этапы курсовой работы."""
    
    print("\n" + "=" * 70)
    print(" КУРСОВАЯ РАБОТА")
    print(" Тема: Применение ансамблевых методов для анализа клиентского оттока")
    print("=" * 70)
    
    # ==================== ЭТАП 1: ЗАГРУЗКА И ПРЕДОБРАБОТКА ====================
    df = load_data()
    display_dataset_info(df)
    
    df_clean = clean_data(df)
    df_clean = encode_target(df_clean)
    
    # Сохраняем копию для EDA (до One-Hot Encoding)
    df_for_eda = df_clean.copy()
    y_for_eda = df_clean['Churn'].copy()
    
    X, y, feature_names = prepare_features(df_clean)
    
    # ==================== ЭТАП 2: ОПИСАТЕЛЬНЫЙ АНАЛИЗ ====================
    run_full_eda(df_for_eda, y_for_eda)
    
    # ==================== ЭТАП 3: РАЗДЕЛЕНИЕ И МАСШТАБИРОВАНИЕ ====================
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    
    # ==================== ЭТАП 4: ОБУЧЕНИЕ БАЗОВЫХ МОДЕЛЕЙ ====================
    print("\n" + "=" * 70)
    print(" ЭТАП 4. ОБУЧЕНИЕ БАЗОВЫХ МОДЕЛЕЙ (7 штук)")
    print("=" * 70)
    
    base_models = get_base_models()
    results_base = train_and_evaluate(
        base_models, X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    print("\n[✓] Результаты базовых моделей:")
    print(results_base.to_string(index=False))
    
    # Сохраняем результаты в CSV
    results_base.to_csv(f"{RESULTS_DIR}/base_models_results.csv", index=False)
    print(f"[OK] Результаты сохранены: {RESULTS_DIR}/base_models_results.csv")
    
    # ==================== ЭТАП 5: ОБУЧЕНИЕ АНСАМБЛЕЙ ====================
    print("\n" + "=" * 70)
    print(" ЭТАП 5. ОБУЧЕНИЕ АНСАМБЛЕВЫХ МОДЕЛЕЙ")
    print("=" * 70)
    
    ensemble_models = get_ensemble_models(base_models)
    results_ensemble = train_and_evaluate(
        ensemble_models, X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    print("\n[✓] Результаты ансамблевых моделей:")
    print(results_ensemble.to_string(index=False))
    
    results_ensemble.to_csv(f"{RESULTS_DIR}/ensemble_models_results.csv", index=False)
    print(f"[OK] Результаты сохранены: {RESULTS_DIR}/ensemble_models_results.csv")
    
    # ==================== ЭТАП 6: СРАВНЕНИЕ И ВИЗУАЛИЗАЦИЯ ====================
    print("\n" + "=" * 70)
    print(" ЭТАП 6. СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ")
    print("=" * 70)
    
    all_results = pd.concat([results_base, results_ensemble], ignore_index=True)
    all_results_sorted = all_results.sort_values(by='ROC-AUC', ascending=False)
    
    print("\n[✓] Итоговая таблица всех моделей (отсортирована по ROC-AUC):")
    print(all_results_sorted.to_string(index=False))
    all_results_sorted.to_csv(f"{RESULTS_DIR}/all_models_results.csv", index=False)
    
    # Определяем лучшую модель до оптимизации
    best_model_name = all_results_sorted.iloc[0]['Model']
    best_model_score = all_results_sorted.iloc[0]['ROC-AUC']
    print(f"\n[★] Лучшая модель до оптимизации: {best_model_name} "
          f"(ROC-AUC = {best_model_score:.4f})")
    
    # Визуализация
    plot_metrics_comparison(all_results)
    
    # Объединяем обученные модели для ROC-кривых
    all_trained_models = {**base_models, **ensemble_models}
    plot_roc_auc_comparison(all_trained_models, X_test_scaled, y_test)
    plot_confusion_matrices(all_trained_models, X_test_scaled, y_test)
    
    # Кривые обучения для лучшей базовой и лучшей ансамблевой
    if 'RandomForest' in base_models:
        plot_learning_curves(base_models['RandomForest'], 
                             X_train_scaled, y_train, 'RandomForest')
    
    # Подробный отчёт для CatBoost
    if 'CatBoost' in ensemble_models:
        print_classification_report(
            ensemble_models['CatBoost'], X_test_scaled, y_test, 'CatBoost'
        )
    
    # ==================== ЭТАП 7: ОПТИМИЗАЦИЯ ====================
    best_catboost, opt_results = optimize_catboost(
        X_train_scaled, y_train, X_test_scaled, y_test, feature_names
    )
    
    # ==================== ИТОГОВЫЙ ОТЧЁТ ====================
    print("\n" + "=" * 70)
    print(" ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 70)
    
    print("\n[★] Лучшая модель ДО оптимизации:")
    print(f"    {best_model_name} — ROC-AUC = {best_model_score:.4f}")
    
    print("\n[★] Оптимизированная модель (CatBoost + GridSearch):")
    print(f"    Лучшие параметры: {opt_results['best_params']}")
    print(f"    ROC-AUC (CV):     {opt_results['best_cv_score']:.4f}")
    print(f"    ROC-AUC (тест):   {opt_results['test_roc_auc']:.4f}")
    print(f"    Recall (тест):    {opt_results['test_recall']:.4f}")
    print(f"    Время поиска:     {opt_results['search_time']:.2f} сек")
    
    improvement = opt_results['test_roc_auc'] - best_model_score
    print(f"\n[✓] Прирост ROC-AUC после оптимизации: {improvement:+.4f}")
    
    print("\n" + "=" * 70)
    print(" ВСЕ ГРАФИКИ СОХРАНЕНЫ В ПАПКУ:", RESULTS_DIR)
    print("=" * 70)
    
    # Список всех сохранённых файлов
    import os
    files = os.listdir(RESULTS_DIR)
    print("\nСодержимое папки results/:")
    for f in sorted(files):
        print(f"  - {f}")
    
    print("\n[✓] Работа завершена успешно!")


if __name__ == "__main__":
    main()