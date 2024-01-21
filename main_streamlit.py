import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import random

from python_modules.interpretation import get_interpretation
from python_modules.ML import load_model_and_predict

st.set_page_config(
    page_title="Oh My App!",
    page_icon="./images/icon.png",
    layout="wide",
)

good_images = ['./images/nice_1.jpg', './images/nice_2.jpg', './images/nice_3.jpg']
sad_images = ['./images/bad_1.jpg', './images/bad_2.jpg']


def started_page():
    intro_1, intro_2 = st.columns([0.8, 0.3])

    with intro_1:
        st.title('Останется ли пассажир доволен перелетом? ✈️')
        st.write("### Для ответа на вопрос предоставте следующие данные о пассажире:")

    with intro_2:
        st.image('./images/flying.gif', output_format='gif', )


def print_features_values():
    """Отрисовка выбора параметров"""
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.radio('#### Пол пассажира: ', ('Мужской 🧑', 'Женский👧'))
        loyal = st.radio('#### Лоялен ли клиент авиакомпании: ', ('Клиент лоялен😀', 'Клиент не лоялен😒'))

    with col2:
        travel_type = st.radio('#### Тип поездки: ', ('Деловая👔', 'Персональная👕'))
        class_ = st.radio('#### Класс обслуживания: ', ('Бизнес💲💲💲', 'Эконом+💲💲', 'Эконом💲'))

    with col3:
        age = st.number_input("#### Возраст пассажира (число полных лет)", step=1, value=25)
        distance = st.number_input("#### Расстояние перелета (км)", value=2000)

    return gender, loyal, travel_type, class_, age, distance


def get_features_to_df(gender, loyal, travel_type, class_, age, distance):
    # производим преобразование
    to_1 = ['Мужской 🧑', 'Клиент лоялен😀', 'Деловая👔']
    dict_class = {
        'Бизнес💲💲💲': 2,
        'Эконом+💲💲': 1,
        'Эконом💲': 0
    }
    gender = 1 if gender in to_1 else 0
    loyal = 1 if loyal in to_1 else 0
    travel_type = 1 if travel_type in to_1 else 0
    class_ = dict_class[class_]
    distance = distance * 0.621371      # перевод в мили

    # создаем X_test
    X_test_x = pd.DataFrame([[gender, age, loyal, travel_type, class_, distance]])
    return X_test_x


def interpretation_result():
    """
    После чтения параметров выделяет похожих клиентов в df
    Смотрит средние оценки похожих (чтобы вывести, чем они довольны а чем нет) по 4м категориям

    :return:
    """
    print('123')
    have_enought_data, features_importance, mean_check, mean_satisfaction = get_interpretation(*X_test.iloc[0])
    if have_enought_data:
        st.write(f"### Среди похожих пассажиров доля довольных: {mean_satisfaction}%")

        st.write(f"### Средние оценки похожих пассажиров и вклад этой оценки в удовлетворенность:")
        for index_x, value_x in zip(features_importance.index, features_importance):
            _, col_index, col_val = st.columns([0.1, 0.35, 0.9])
            with col_index: st.write(f"#### {index_x}")
            with col_val: st.write(f"#### {round(mean_check[index_x], 1)} -- ({round(value_x, 1)} %)")

    else:
        st.write("О подобных пассажирах слишком мало известно, чтобы сделать выводы.")


if __name__ == "__main__":
    started_page()
    features = print_features_values()  # отрисовка выбора параметров
    X_test = get_features_to_df(*features)  # преобразование выбранных параметров к исходных данных (male=0, female=1)

    # предсказываем
    col_1, col_2, col_3, col_4 = st.columns([0.6, 0.01, 0.16, 0.35])
    img_col, interp_col = st.columns([0.6, 1.5])

    with col_3: go = st.button('#### Предсказать')

    result = 'dont_know'
    if go:
        # загрузка маленькой модели и вывод результата: доволен или нет
        predict = load_model_and_predict(X_test, path='./data/little_model_weights.mv')
        result = 'success' if predict == 'satisfied' else 'unsuccess'

    with col_1:
        # отрисовываем результат
        # успех
        if result == 'success':
            st.success('## Пассажир останется довольным!😄👍')
            with img_col: st.image(random.choice(good_images), width=300, output_format='jpg')
            with interp_col: interpretation_result()     # вывод интерпретации результата
        # недовольство
        elif result == 'unsuccess':
            st.warning("## Пассажир будет недоволен.🙁👎")
            with img_col:  st.image(random.choice(sad_images), width=300, output_format='jpg')
            with interp_col: interpretation_result()     # вывод интерпретации результата

        #
        elif result == 'dont_know':
            st.info('## Для предсказания нажми на кнопку 🤷‍♂️👉',)
            with img_col: st.image('./images/think_2.png', width=300)
            interp = False
