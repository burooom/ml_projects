# Определение возраста покупателей

[HTML](https://github.com/burooom/ml_projects/tree/main/Computer_vision-regression/DL_School_projects/Journey_to_Springfield.html)     [ipynb](https://github.com/burooom/ml_projects/tree/main/DL_School_projects/Journey_to_Springfield.ipynb)

## Цели исследования
 Нужно помочь телекомпании FOX в обработке их контента. Как вы знаете сериал Симсоны идет на телеэкранах более 25 лет и за это время скопилось очень много видео материала. Персоонажи менялись вместе с изменяющимися графическими технологиями и Гомер 2018 не очень похож на Гомера 1989.
 Задачей будет научиться классифицировать персонажей проживающих в Спрингфилде.

 [Страница соревнования на kaggle](https://www.kaggle.com/competitions/journey-springfield)

## Результаты исследования
- Проведено обучение сверточной нейросетевой модели на хребте модели ResNet-200 с добавлением полносвязных регрессионных слоев, активируемых Mish функцией активации.
- Цель исследования достигнута: получены отличные показатели качества модели по метрике F1 - 0.99893, что позволило занять 78-е место из 4457 участников.

## Стек технологий
- python
- pandas
- torch
- numpy
- pillow
- torchvision
- sklearn.preprocessing
- OpenCV
- ResNet
