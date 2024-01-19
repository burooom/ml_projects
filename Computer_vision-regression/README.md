# Определение возраста покупателей

[HTML](https://github.com/burooom/yp_ml_projects/tree/main/Computer_vision-regression/Computer_vision-regression.html)     [ipynb](https://github.com/burooom/yp_ml_projects/tree/main/Computer_vision-regression/Computer_vision-regression.ipynb)

## Цели исследования
Для супермаркета «Хлеб-Соль» подготовить нейросетевую модель для системы компьютерного зрения. Задача супермаркета состоит в определении возраста клиентов по фотофиксации в прикассовой зоне.

По набору данных <a href='https://chalearnlap.cvc.uab.cat/dataset/26/description/'>APPA-REAL</a> необходимо построить модель, которая по фотографии определит приблизительный возраст человека.

По каждой квартире на продажу доступны два вида данных. Первые вписаны пользователем, вторые — получены автоматически на основе картографических данных. Например, расстояние до центра, аэропорта, ближайшего парка и водоёма.
## Результаты исследования
- В стороннем тренажере проведено обучение сверточной нейросетевой модели на хребте модели ResNet-50 с добавлением полносвязных регрессионных слоев, активируемых ReLU.
- Цель исследования достигнута: получены удовлетворительные результаты модели по метрике МАЕ (меньше 8)

## Стек технологий
python
pandas
numpy
tensorflow.keras.preprocessing.image.ImageDataGenerator
tensorflow.keras.layers.GlobalAveragePooling2D, Dense
tensorflow.keras.models.Sequential
tensorflow.keras.preprocessing.image.ImageDataGenerator
tensorflow.keras.optimizers.Adam
tensorflow.keras.applications.resnet.ResNet50


