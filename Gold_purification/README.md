# Исследование технологического процесса очистки золота

[HTML](https://github.com/burooom/yp_ml_projects/tree/main/Gold_purification/Gold_purification.html)     [ipynb](https://github.com/burooom/yp_ml_projects/tree/main/Gold_purification/Gold_purification.ipynb)

## Цели исследования
По данным с параметрами добычи и очистки руд подготовить прототип модели машинного обучения для компании «Цифры», которая разрабатывает решения для работы промышленных предприятий.
Модель должна предсказать коэффициент восстановления золота из золотосодержащей руды. Модель поможет оптимизировать производство, чтобы не запускать предприятие с убыточными характеристиками.

## Результаты исследования

В ходе работы были проанализированы и обработаны предоставленные данные и подготовлена модель машинного обучения, способная на обучающей выборке выдавать прогнозный коэффициент восстановления золота лучше baseline модели (Dummy Regressor).
В итоге, удалось обучить модель машинного обучения, которая по метрике sMAPE лучше альтернатив, включая константную baseline модель.

## Стек технологий
python
pandas
numpy
matplotlib
seaborn
sklearn.model_selection.cross_val_score
sklearn.model_selection.GridSearchCV
sklearn.model_selection.RandomizedSearchCV
sklearn.metrics.mean_absolute_error
sklearn.metrics.make_scorer
sklearn.preprocessing.StandardScaler
sklearn.dummy.DummyRegressor
sklearn.tree.DecisionTreeRegressor
sklearn.ensemble.RandomForestRegressor
sklearn.linear_model.LinearRegression
