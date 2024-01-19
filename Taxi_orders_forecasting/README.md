# Прогнозирование заказов такси
[HTML](https://github.com/burooom/yp_ml_projects/tree/main/Taxi_orders_forecasting/Taxi_orders_forecasting.html)     [ipynb](https://github.com/burooom/yp_ml_projects/tree/main/Taxi_orders_forecasting/Taxi_orders_forecasting.ipynb)

## Цели исследования
По историческим данным о заказах такси в аэропортах построить для заказчика «Чётенькое такси» модель предсказания количества заказов на следующий час с целью привлечения водителей в пиковые часы нагрузок.

Заказчику важно значение метрики RMSE на тестовой выборке, не превышающее 48 заказов.

## Результаты исследования
- Для достижения целей данные были ресемплированы до частоты в 1 час
- Временной ряд нестационарен, поэтому для его анализа применена техника декомпозиции, которая позволила выявить признаки, помогающие моделям эффективнее обучаться
- Методом кросс-валидации были обучены 4 регрессионные модели (Линейная регрессия, Случайный лес, LightGBM, CatBoost)
- Проведено тестирование лучшей модели по метрике RMSE, полученый результаты, удовлетворяющие критерию (`важно значение метрики RMSE на тестовой выборке, не превышающее 48 заказов`)
- Заказчику рекомендавана полученная модель

## Стек технологий
- python
- pandas
- numpy
- matplotlib
- sklearn.model_selection.GridSearchCV
- sklearn.model_selection.RandomizedSearchCV
- sklearn.model_selection.train_test_split
- sklearn.model_selection.TimeSeriesSplit
- sklearn.metrics.mean_squared_error
- sklearn.preprocessing.StandardScaler
- sklearn.preprocessing.PolynomialFeatures
- sklearn.ensemble.RandomForestRegressor
- sklearn.linear_model.LinearRegression
- catboost.CatBoostRegressor
- lightgbm.LGBMRegressor
- lightgbm.plot_importance
- statsmodels.tsa.seasonal.seasonal_decompose

