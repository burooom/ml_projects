# Обучение модели классификации комментариев

[HTML](https://github.com/burooom/ml_projects/tree/main/Yandex.Practicum_projects/Comments_classification/Comments_classification.html)     [ipynb](https://github.com/burooom/ml_projects/tree/main/Yandex.Practicum_projects/Comments_classification/Comments_classification.ipynb)

## Цели исследования
Для предприятия «Викишоп» с целью автоматической классификации комментариев на позитивные и негативные обучить ML модель по набору размеченных данных в виде корпуса текстов, классифицированных на токсичные и нетоксичные. Построить модель со значением метрики качества F1 не меньше 0.75

## Результаты исследования
- Текст был очищен от несловарных символов, затем лемматизирован и векторизован
- В процессе обучения моделей машинного обучения для некоторых из них были получены удовлетворительные характеристики по F1 мере.
- В ходе тестирования лучшей отобранной модели подтвердилось качество обученной модели на тестовой выборке, модель рекомендована заказчику

## Стек технологий
- python
- pandas
- numpy
- re
- sklearn.model_selection.GridSearchCV
- sklearn.model_selection.RandomizedSearchCV
- sklearn.model_selection.train_test_split
- sklearn.model_selection.StratifiedKFold
- sklearn.metrics.accuracy_score
- sklearn.metrics.f1_score
- sklearn.metrics.roc_auc_score
- sklearn.metrics.roc_curve
- sklearn.linear_model.LogisticRegression
- sklearn.linear_model.SGDClassifier
- sklearn.feature_extraction.text.TfidfVectorizer
- sklearn.feature_extraction.CountVectorizer
- sklearn.feature_selection.RFE
- sklearn.feature_selection.RFECV
- nltk
- nltk.corpus.stopwords
- nltk.stem.WordNetLemmatizer
- nltk.tokenize.WhitespaceTokenizer
- nltk.corpus.wordnet
- scipy.stats
- scipy.sparse
- catboost.CatBoostClassifier
- lightgbm.LGBMClassifier
- lightgbm.plot_importance

## TODO
Перевести обучение на sklearn.pipeline.Pipeline, чтобы избавиться от возможной утечки данных на кроссвалидации
