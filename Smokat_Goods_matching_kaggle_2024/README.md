# Исследование матчинга товаров

[HTML](https://github.com/burooom/yp_ml_projects/tree/main/Goods_matching/Goods_matching.html)     [ipynb](https://github.com/burooom/yp_ml_projects/tree/main/Goods_matching/Goods_matching.ipynb)

## Цели исследования
Заказчик предложил решить задачу по матчингу товаров: Предстоит реализовать финальную часть пайплайна матчинга. В ней следует принять решение для каждой пары (товар предлагаемый продавцом — товар на площадке), является ли она матчем или нет (бинарная классификация). Для этого у каждой пары есть набор признаков и наборы векторов (картиночные и текстовые), которые описывают товары из этой пары.

В качестве метрики качества решения используется F-score.

[Страница задания на Kaggle](https://www.kaggle.com/competitions/binary-classification-offers-on-the-marketplace/overview)

## Результаты исследования
- Для предстказания "матча" создана модель на базе ансамбля моделей линейной регрессии, случайного леса и градиентного бустинга, позволяющую идентифицировать матч (совпадение) между товарами, представленными определенныими признаками, а также текстовыми и графическими эмбеддингами
- В ходе работ проведена нетривиальная предобработка данных: используя машинное обучение и признаки эмбеддингов, сжатые методом основных компонент, восстановлены пропуски в данных и скорректированы нереальные значения цен
- Для итоговой ML-модели получены показатели качества ~0.9 по f1-мере

## Стек технологий
- python
- pandas
- numpy
- matplotlib
- lightgbm
- sklearn.base.TransformerMixin
- sklearn.base.BaseEstimator
- sklearn.compose.ColumnTransformer
- sklearn.decomposition.PCA
- sklearn.discriminant_analysis.LinearDiscriminantAnalysis
- sklearn.ensemble.VotingClassifier
- sklearn.linear_model.LogisticRegression
- sklearn.metrics.f1_score
- sklearn.model_selection.train_test_split
- sklearn.model_selection.StratifiedKFold
- sklearn.model_selection.HalvingGridSearchCV
- sklearn.pipeline.Pipeline
- sklearn.preprocessing.StandardScaler
- sklearn.preprocessing.OneHotEncoder
- sklearn.preprocessing.SplineTransformer
- skopt.BayesSearchCV


