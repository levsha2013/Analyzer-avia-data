import pandas as pd
import streamlit as st
from ML import load_model_and_predict

st.title('Анализ данных о полете пассажиров.')

st.write("##### Предсказание, останется ли доволен пассажир полетом"
         "на основании болшого количества параметров.")

col1, col2, col3 = st.columns(3)
with col1:
    gender = st.radio('Выберите пол пассажира: ', ('Мужской', 'Женский'))
    loyal = st.radio('Лоялен ли клиент авиакомпании: ', ('Клиент лоялен', 'Клиент не лоялен'))

with col2:
    travel_type = st.radio('Выберите тип поездки: ', ('Деловая', 'Персональная'))
    class_ = st.radio('Выберите класс обслуживания: ', ('Бизнес', 'Эконом+', 'Эконом'))

with col3:
    age = st.number_input("Введите возраст пассажира (число полных лет)", step=1)
    distance = st.number_input("Укажите расстояние перелета (км)")

st.write("##### Учет оценок различных факторов")
col_1, col_2, col_3 = st.columns(3)

with col_1:
    wifi = st.select_slider("Качество wifi", [0,1,2,3,4,5], value=5)
    time_arrive = st.select_slider("Удобство времени прилета", [0,1,2,3,4,5], value=5)
    bron = st.select_slider("Удобство бронирования билета", [0, 1, 2, 3, 4, 5], value=5)
    place_in = st.select_slider("Расположение выхода на посадку", [0, 1, 2, 3, 4, 5], value=5)
    eat = st.select_slider("Качество еды на борту", [0, 1, 2, 3, 4, 5], value=5)


with col_2:
    place_into = st.select_slider("Удобство выбора места в самолете", [0, 1, 2, 3, 4, 5], value=5)
    seat_comfort = st.select_slider("Удобство сиденья в самолете", [0, 1, 2, 3, 4, 5], value=5)
    fun = st.select_slider("Качество развлечений", [0, 1, 2, 3, 4, 5], value=5)
    service = st.select_slider("Качество обслуживания", [0, 1, 2, 3, 4, 5], value=5)
    bot_place = st.select_slider("Оценка места в ногах", [0, 1, 2, 3, 4, 5], value=5)


with col_3:
    bag = st.select_slider("Качество обращения с багажом", [0, 1, 2, 3, 4, 5], value=5)
    registr = st.select_slider("Оценка регистрации", [0, 1, 2, 3, 4, 5], value=5)
    service_2 = st.select_slider("Качество обслуживания_2", [0, 1, 2, 3, 4, 5], value=5)
    clean = st.select_slider("Оценка чистоты", [0, 1, 2, 3, 4, 5], value=5)


# производим преобразование
to_1 = ['Мужской', 'Клиент лоялен', 'Деловая']
dict_class = {
    'Бизнес': 2,
    'Эконом+': 1,
    'Эконом': 0
}
gender = 1 if gender in to_1 else 0
loyal = 1 if loyal in to_1 else 0
travel_type = 1 if travel_type in to_1 else 0
class_ = dict_class[class_]
distance = distance * 0.621371  # перевод в мили

# создаем X_test
X_test = pd.DataFrame([[gender, age, loyal, travel_type, class_, distance,
                        wifi, time_arrive, bron, place_in, eat,
                        place_into, seat_comfort, fun, service, bot_place,
                        bag, registr, service_2, clean]])

# предсказываем
go = st.button('Предсказать удовлетворенность пассажира')
if go:
    predict = load_model_and_predict(X_test, path='./data/model_weights.mv')

    if predict == 'satisfied':
        st.write('Пассажир останется довольным!')
    else:
        st.write("К сожалению, пассажир не оценит перелет положительно.")