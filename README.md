# Pet project: analyzer avia_data
### Данный проект был создан как выпускная работа курса ["Разработка ML сервиса: от идеи к прототипу"](https://stepik.org/course/176820/info)  (создан [преподавателями магистратуры ВШЭ](https://www.hse.ru/org/persons/211268525))
![Пример отображения работающего сервиса](https://github.com/levsha2013/avia-ancet-data-analyser-service/blob/master/images/main_print.png)

### Содержание репозитория:
- Jupyter:
  - pilot_preprocessing.ipynb (предобработка данных)
  - Factor analyzer.ipynb (факторный анализ предобработанных данных)
  - learn model.ipynb (обучение модели с подбором гиперпараметров)
- data (исходные и преобразованные данные в формате CSV)
- images (изображения для сервиса)
- python_modules (модули python, к которым обращается запускаемым сервис)

### [Входные данные](https://github.com/levsha2013/Analyzer-avia-data/blob/master/data/pilot.csv) содержат следующие группы признаков:
- [X] характеристики пассажиров (Age, Gender, Type_of_Travel, ...)
- [X] оценки от пассажиров полета (Food and drink, Seat comfort, Cleanliness, ...)
- [X] целевая переменная - доволен пассажир перелетом или нет (satifaction)

### Для построения серсиса было проведено следующее:
- ✅ обнаружение и устранение ошибок в данных
- ✅ стат. обоснованность факторного анализа и непосредственно факторный анализ (осмысленное уменьшение размерности)
- ✅ подбор гиперпараметров с помощью optuna для [CatBoostClassifier](https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier) по f1 метрике на GPU
- ✅ написание сервиса на streamlit для взаимодействия с обученной моделью и данными
- ✅ все фотографии довольных/недовольных пассажиров созданы моделью [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5?text=sad+airplane+passenger+cry)

***Код каждого из шагов с комментариями находится в каталоге Jupyter***

### Для локального запуска проекта необходимо:
1. Скачать репозиторий.
2. Выполнить низлежащий код:
```python
py -3.10 -m venv venv           # создаем виртуальное окружение
.\venv\Scripts\activate.bat     # активация виртуальной среды
pip install -r requirements.txt # установка зависимостей из файла requirenments.txt
streamlit run main_streamlit.py # запуск сервиса
```

## Результаты работы

1. Грамотный EDA и предобработка данных.
2. Выдвижение и подтверждение гипотезы о разумности факторного анализа (14 признаков удалось свести к четким 4м факторам).
3. В качестве предсказания была использована модель CatBoostClassifier с подбором гиперпараметров через opruna. Достигнутое значение метрики f_score 0.88.
4. Реализован подход выделения похожих пассажиров, нахождение их средних оценок и важности каждой из них (модуль interpretation.py).

