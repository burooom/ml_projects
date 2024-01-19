# Определение выгодного тарифа для телеком компании

[HTML](https://github.com/burooom/yp_ml_projects/tree/main/Telecom/Telecom.html)     [ipynb](https://github.com/burooom/yp_ml_projects/tree/main/Telecom/Telecom.ipynb)

## Цели исследования
Провести анализ и обработку клиентских данных сотового оператора мобильной связи «Мегалайн» по данным клиентов, перешедших на новые тарифы с целью:

- Построения системы, способной проанализировать поведение клиентов
- Предложения пользователям нового тарифа: «Смарт» или «Ультра»

## Результаты исследования

В соответствии с целями работы удалось построить предсказательные модели, способные проанализировать поведение клиентов и предложить пользователям новые тарифы «Смарт» или «Ультра» с показателем accuracy на тестовых данных большим, чем 0,75.

## Стек технологий
python
pandas
numpy
matplotlib
sklearn.linear_model.LogisticRegression
sklearn.tree.DecisionTreeClassifier
sklearn.ensemble.RandomForestClassifier
sklearn.dummy.DummyClassifier
sklearn.metrics.f1_score
sklearn.metrics.mean_squared_error
sklearn.metrics.accuracy_score
sklearn.model_selection.train_test_split
sklearn.model_selection.ParameterSampler
sklearn.model_selection.GridSearchCV
sklearn.model_selection.StratifiedShuffleSplit
xgboost.XGBClassifier
