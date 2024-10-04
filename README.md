# Классификация изображений для сервиса DonorSearch.


## Описание
Заказчик- DonorSearch- занимается развитием донорства. Для этого есть платформа DonorSearch.org, где для доноров доступны бонусная программа, игрофикация пути донора и многое другое. Важной является проверка честности доноров и корректности внесенных донаций. Подтверждение производится по справке установленной формы (№405), такую справку донор получает в центре крови и загружает как картинку в личный кабинет. На следующем этапе  с помощью сервиса OCR (optical character recognition) происходит распознавание табличной информации на бланке справки и записи результатов в .csv файл. Существующая версия сервиса требует вертикальной ориентации справки.

### Цель

Целью данного проекта является разработать модель определения ориентации справки и автоматического поворота ее в нормальное положение перед запуском сервиса OCR (optical character recognition).

### **Инструкции по использованию:**
Для того, чтобы корректно работать с данным проектом, необходимо:
```
- Для классификации изображений: выгрузить файлы data_preparation.ipynb, rotation_angle_detection.ipynb, own_functions.py, requirements.txt

- Для запуска приложения через Docker:выгрузить requirements.txt, app.py, Dockerfile.

```

### Исходные данные

 Датасет из 173 справок различного формата, включая справку 405 с таблицей.

### Метрика и условия: 
- Метрика оценки модели для многоклассовой классификации– Accuracy.
- Построены ROC-кривые и матрицы ошибок для наилучшей из моделей


## Выбор модели

В проекте использовалось:
- предобученная модель ResNet50
- предобученная модель VGG11

**Результат:**

Создание микросервиса для последующей интеграции в продукт заказчика. 

## Структура репозитория:

| #    | Наименование файла                | Описание   |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1.   | [README.md](https://github.com/IliaShi/donor_search/blob/main/README.md) | Представлена основная информация по проекту и его результатах   |
| 2.   | [rotation_angle_detection.ipynb](https://github.com/IliaShi/donor_search/blob/main/rotation_angle_detection.ipynb) | Тетрадка с основным решением |
| 3.   | [data_preparation.ipynb](https://github.com/IliaShi/donor_search/blob/main/data_preparation.ipynb) | Тетрадка для подготовки датасета из исходных фотографий   |
| 4.   | [experiments.ipynb](https://github.com/IliaShi/donor_search/blob/main/experiments.ipynb) | Тетрадка с экспериментами и поиском решения   |
| 5.   | [own_functions.py](https://github.com/IliaShi/donor_search/blob/main/own_functions.py) | Собственные функции для файла rotation_angle_detection  |
| 6.   | [requirements.txt](https://github.com/IliaShi/donor_search/blob/main/requirements.txt) | Список всех библиотек и их версии, необходимых для установки в виртуальной среде для запуска кода проекта   |
| 7.   | [Dockerfile](https://github.com/IliaShi/donor_search/blob/main/app/Dockerfile) | Докер-файл для запуска приложения
| 8.   | [requirements.txt](https://github.com/IliaShi/donor_search/blob/main/app/requirements.txt) | Список всех библиотек и их версии, необходимых для установки в виртуальной среде для запуска приложения в Докере |
| 9.   | [app.py](https://github.com/IliaShi/donor_search/blob/main/app/scr/app.py) |Скрипт для запуска приложения |
|10.   | [model.py](https://github.com/IliaShi/donor_search/blob/main/app/scr/app.py) |Запуск сохраненной модели лучшего решения |


## Итоги

**Основные выводы:**  
1. Для обучения использован датасет, предоставленный заказчиком (173 изображения). Количество изображений увеличено в 4 раза за счет поворта на 90, 180, 270 градусов каждого исходного изображения.
2. На исходных изображения от заказчика с помощью графичекого редктора удалены артефакты в виде белых прямоугольников, скрывающих персональные данные (экспериментально установлено, что данные артефакты снижают точность модели).
3. Обучены две модели: на основе ResNet-50 и VGG-11 с предобученными весами. Три последних слоя моделей разморожены и дообучены. 
4. Обучение на цветных изображениях 128х128, размер батча 64, эпох 10.
5. Обе модели показали высокие значения accuracy на валидационной выборке: ResNet - 0.98, VGG - 1.0. Инференс моделей при этом был сопоставим. Выбрана модель VGG за счет более высокой метрики качества.
6. Точность модели на основе VGG проверена на тестовой выборке: accuracy - 0.98, AUC - 0.99. Ошибки модели зафиксированы для изображений с углом поворта 0 и 90 градусов.
7. Разработан функционирующий микросервис на базе FastAPI, позволяющий определять угол поворта загруженной справки и возвращать правильно ориентированное изображение.

**Проект может быть развит в следующих направлениях:**
  
   * **Обучение моделей на датасете большего размера:**  Обучение моделей на большем датасете может улучшить их качество и обобщающую способность.
   * **Обучение моделей на датасете без скрытия личных данных:** Так как на данных изображениях личные данные были скрыты, а на реальных справках данные остаются, было предположено, что это влияет на качество модели.


## Cтатус: 
Завершён.

## Стэк:
- PyTorch - для работы с Deep Learning моделями
- Numpy для математических операций
- matplotlib и seaborn - для визуализации данных.
- Pillow, Open-CV для работы с изображениями
- Scikit-learn для расчета метрик и обработки результатов
- FastApi для создания интерфейса модели
- Docker для создания микросервиса

## Команда проекта
- [Илья Широких](https://github.com/IliaShi)
- [Гульшат Зарипова](https://github.com/gulshart)

