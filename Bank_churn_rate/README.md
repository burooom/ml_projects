# Прогнозирование оттока клиентов Банка

[HTML](https://github.com/burooom/yp_ml_projects/tree/main/Bank_churn_rate/Bank_churn_rate.html)     [ipynb](https://github.com/burooom/yp_ml_projects/tree/main/Bank_churn_rate/Bank_churn_rate.ipynb)

## Цели исследования
Построить модели прогнозирования, способные предсказывать, уйдёт клиент из банка в ближайшее время или нет.
## Результаты исследования

В ходе исследования данных и построения моделей прогнозирования были получены следующие результаты:

- Показано, что балансировка несбалансированных данных как правило улучшает оценки модели. Обнаружено, что разные методы балансировки по-разному влияют на результат в зависимости от типа используемой ML-модели. В ходе исследования была выявлена более высокая чувствительность метрики f1 (относительно ROC-AUC) к дисбалансу классов целевой переменной.
- Цели работы достигнуты: построена ML-модель, способная предсказывать отток клиентов с большим показателем f1-меры, чем установленны заказчиком минимум.

## Стек технологий
- pandas
- numpy
- matplotlib
- tqdm
- sklearn.linear_model.LogisticRegression
- sklearn.tree.DecisionTreeClassifier
- sklearn.ensemble.RandomForestClassifier
- sklearn.ensemble.VotingClassifier
- sklearn.utils.shuffle
- sklearn.preprocessing.StandardScaler
- sklearn.preprocessing.PolynomialFeatures
- sklearn.preprocessing.OneHotEncoder
- sklearn.preprocessing.SplineTransformer
- sklearn.metrics.f1_score
- sklearn.metrics.roc_auc_score
- sklearn.metrics.roc_curve
- sklearn.model_selection.train_test_split
- sklearn.model_selection.ParameterSampler
- xgboost.XGBClassifier
