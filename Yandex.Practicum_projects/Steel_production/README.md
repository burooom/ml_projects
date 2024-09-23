# Промышленность: обработка стали

[HTML](https://github.com/burooom/ml_projects/tree/main/Yandex.Practicum_projects/Steel_production/Steel_production.html)     [ipynb](https://github.com/burooom/ml_projects/tree/main/Yandex.Practicum_projects/Steel_production/Steel_production.ipynb)

## Цели исследования
Необходимо построить модель, которая предскажет температуру стали.

## Результаты исследования
Данные были описаны, проанализированы, очищены от аномалий и выбросов, выделены признаки, необходимые для обучения.

Для обучения были выбраны 4 регрессионные модели:
1. Константная baseline модель (DummyRegressor, sklearn), предсказывающая целевой признак по среднему значению;
2. Линейная регрессия (LinearRegressor, sklearn);
3. Случайный лес (RandomForest, sklearn);
4. Градиентный бустинг (LGBMRegressor, lightgbm)

- Проведено обучение моделей машинного обучения и протестирована выбранная модель на предмет достижения поставленной цели по качеству предсказания
- По результатам проведенного методом кросс-валидации обучения была отобрана регрессионная модель, показавшая лучшие результаты.

## Стек технологий
- python
- pandas
- numpy
- matplotlib
- seaborn
- scipy.stats
- sklearn.model_selection.cross_val_score
- sklearn.model_selection.GridSearchCV
- sklearn.model_selection.RandomizedSearchCV
- sklearn.model_selection.train_test_split
- sklearn.metrics.mean_absolute_error
- sklearn.preprocessing.StandardScaler
- sklearn.preprocessing.PolynomialFeatures
- sklearn.preprocessing.SplineTransformer
- sklearn.pipeline.Pipeline
- sklearn.dummy.DummyRegressor
- sklearn.ensemble.RandomForestRegressor
- sklearn.ensemble.IsolationForest
- sklearn.linear_model.LinearRegression
- lightgbm.LGBMRegressor
- lightgbm.plot_importance
