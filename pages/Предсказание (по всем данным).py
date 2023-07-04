import pandas as pd
import streamlit as st
from ML import load_model_and_predict


st.set_page_config(
    page_title="Oh My App!",
    page_icon="./images/icon.png",
    layout="wide"
)


st.title('Останется ли пассажир доволен перелетом? ✈️')


st.write("##### Предсказание, останется ли доволен пассажир полетом"
         "на основании болшого количества параметров.")

col1, col2, col3 = st.columns(3)
with col1:
    gender = st.radio('Выберите пол пассажира: ', ('Мужской 🧑', 'Женский👧'))
    loyal = st.radio('Лоялен ли клиент авиакомпании: ', ('Клиент лоялен😀', 'Клиент не лоялен😒'))
with col2:
    travel_type = st.radio('Выберите тип поездки: ', ('Деловая👔', 'Персональная👕'))
    class_ = st.radio('Выберите класс обслуживания: ', ('Бизнес💲💲💲', 'Эконом+💲💲', 'Эконом💲'))

with col3:
    age = st.number_input("Введите возраст пассажира (число полных лет)", step=1, value=25)
    distance = st.number_input("Укажите расстояние перелета (км)", value=2000)

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
distance = distance * 0.621371  # перевод в мили

# создаем X_test
X_test = pd.DataFrame([[gender, age, loyal, travel_type, class_, distance,
                        wifi, time_arrive, bron, place_in, eat,
                        place_into, seat_comfort, fun, service, bot_place,
                        bag, registr, service_2, clean]])

# предсказываем
col_result, _, button_pred, _ = st.columns([0.6,0.01,0.16,0.35])

img_col, interp_col = st.columns([0.6,1])
with button_pred:
    go = st.button('Предсказать')

result = 'dont_know'
if go:
    predict = load_model_and_predict(X_test, path='./data/model_weights.mv')
    if predict == 'satisfied':
        result = 'success'
    else:
        result = 'unsuccess'

with col_result:
    # отрисовываем результат
    # успех
    if result == 'success':
        st.success('Пассажир останется довольным!😄👍')
        with img_col:
            st.image('./images/success.jpg', width=300, output_format='jpg')
        with interp_col:
            with st.expander("Интерпретация результата"):
                st.write("Тут будет интерпретация успеха")

    # недовольство
    elif result == 'unsuccess':
        st.warning("К сожалению, пассажир не оценит перелет положительно.🙁👎")
        with img_col:
            st.image('./images/unsuccess.png', width=350)
        with interp_col:
            with st.expander("Интерпретация результата"):
                st.write("Тут будет интерпретация недовольства")

    #
    elif result == 'dont_know':
        st.info('Нажми на кнопку "Предсказать" 🤷‍♂️👉',)
        with img_col:
            st.image('./images/think_2.png', width=300)
        interp = False