# Определение наиболее выгодного региона нефтедобычи

[HTML](https://github.com/burooom/yp_ml_projects/tree/main/Oil_extraction/Oil_extraction.html)     [ipynb](https://github.com/burooom/yp_ml_projects/tree/main/Oil_extraction/Oil_extraction.ipynb)

## Цели исследования
По анализу данных о пробах нефти в трёх регионах (0, 1, 2) необходимо дать рекомендации, в каком регионе разрабатывать новые месторождения для получения наибольшей прибыли для добывающей компании «ГлавРосГосНефть»
Для этого нужно построить модель для определения региона, где добыча принесёт наибольшую прибыль, а также проанализировать возможную прибыль и риски техникой Bootstrap.

## Результаты исследования

- Выяснено, что разработка случайных месторождений в среднем убыточна во всех регионах, выбор сделан в пользу отбора региона, в котором прибыльна разработка "лучших 200" (отобранных моделью) месторождений.
- Определены прибыльные регионы нефтедобычи и уровень риска, определяемый как вероятность того, что разработка будет неприбыльной.
- Рекомендован к разработке регион с максимальной прибылью и уровнем риска меньше критического.

## Стек технологий
python
pandas
numpy
matplotlib
sklearn.model_selection.train_test_split
sklearn.model_selection.cross_val_score
sklearn.linear_model.LinearRegression
sklearn.preprocessing.StandardScaler
sklearn.preprocessing.PolynomialFeatures
sklearn.metrics.mean_squared_error
