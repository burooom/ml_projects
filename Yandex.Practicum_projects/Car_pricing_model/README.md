# Прогнозирование оттока клиентов Банка

[HTML](https://github.com/burooom/ml_projects/tree/main/Yandex.Practicum_projects/Car_pricing_model/Car_pricing_model.html)     [ipynb](https://github.com/burooom/ml_projects/tree/main/Yandex.Practicum_projects/Car_pricing_model/Car_pricing_model.ipynb)

## Цели исследования
По историческим данным (техническим характеристикам, комплектации и ценам автомобилей) построить для заказчика модель определения стоимости автомобиля для использовании в приложении сервиса по продаже автомобилей с пробегом «Не бит, не крашен».

Заказчику важны:

 - качество предсказания;
 - скорость предсказания;
 - время обучения.

Следует обучить и подобрать модель с лучшими результатами по этим характеристикам.

## Результаты исследования

- Построены и протестированы несколько моделей машинного обучения: Линейная регрессия, Случайный лес, Градиентный бустинг (CatBoost и LightGBM).
- По критериям качества предсказания (`(val) RMSE`); скорости предсказания (`Inference time`); времени обучения (`Training time`) была сведена рейтинговая таблица моделей, по которой клиент может посчитать рейтинг нужной ему модели сообразно весам, которые он присвоит критериям. Клиенту был предоставлен механизм оценки обученных моделей на основе вектора его предпочтений.

## Стек технологий
- python
- pandas
- numpy
- matplotlib
- sklearn.model_selection.cross_val_score
- sklearn.model_selection.RandomizedSearchCV
- sklearn.model_selection.train_test_split
- sklearn.metrics.mean_squared_error
- sklearn.preprocessing.StandardScaler
- sklearn.preprocessing.PolynomialFeatures
- sklearn.preprocessing.OneHotEncoder
- sklearn.preprocessing.OrdinalEncoder
- sklearn.ensemble.RandomForestRegressor
- sklearn.linear_model.LinearRegression
- catboost.CatBoostRegressor
- lightgbm.LGBMRegressor
- time.time
