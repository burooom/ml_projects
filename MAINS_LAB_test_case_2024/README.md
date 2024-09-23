# Страхование; предсказание количества визитов клиентов

[HTML](https://github.com/burooom/ml_projects/tree/main/MAINS_LAB_test_case_2024/Clinic_visits_forecast_mains_lab.html)     [ipynb](https://github.com/burooom/ml_projects/tree/main/MAINS_LAB_test_case_2024/Clinic_visits_forecast_mains_lab.ipynb)

[Результаты предсказанные моделью на тестовой выборке](https://github.com/burooom/ml_projects/tree/main/MAINS_LAB_test_case_2024/nov_23_test_results.csv)

[Результаты предсказанные моделью на тестовой выборке, а затем округленные](https://github.com/burooom/ml_projects/tree/main/MAINS_LAB_test_case_2024/nov_23_rounded_test_results.csv)

## Цели исследования
Необходимо построить модель, которая предскажет параметр количества посещений клиники в 2023-м году (`number_of_visits_23`) на основе данных о покупателях страховок.

## Результаты исследования
Данные были описаны и проанализированы. Выделены и созданы признаки, необходимые для обучения.

Для обучения были выбраны 4 регрессионные модели:
1. Константная baseline модель (DummyRegressor, sklearn), предсказывающая целевой признак по среднему значению;
2. Линейная регрессия (LinearRegressor, sklearn);
3. Случайный лес (RandomForest, sklearn);
4. Градиентный бустинг (LGBMRegressor, lightgbm)

- Проведено обучение моделей машинного обучения и протестирована выбранная модель
- По результатам проведенного методом кросс-валидации обучения была отобрана регрессионная модель, показавшая лучшие результаты, ей оказалась модель градиентного бустинга LightGBM с резултатом в 21.8 по метрике MSE

## Стек технологий
- python
- pandas
- numpy
- matplotlib
- lightgbm.LGBMRegressor
- sklearn.model_selection.HalvingGridSearchCV
- sklearn.model_selection.train_test_split
- sklearn.metrics.mean_absolute_error
- sklearn.preprocessing.StandardScaler
- sklearn.preprocessing.PolynomialFeatures
- sklearn.preprocessing.SplineTransformer
- sklearn.pipeline.Pipeline
- sklearn.dummy.DummyRegressor
- sklearn.ensemble.RandomForestRegressor
- sklearn.linear_model.LinearRegression
- skopt.BayesSearchCV


