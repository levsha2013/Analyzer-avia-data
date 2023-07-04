import pandas as pd
import streamlit as st
from ML import load_model_and_predict


st.set_page_config(
    page_title="Oh My App!",
    page_icon="./images/icon.png",
    layout="wide"
)

st.title('Останется ли пассажир доволен перелетом? ✈️')

st.write("Для удобства ввода модель использует только основные параметры.")
st.write("### Для ответа на вопрос предоставте следующие данные о пассажире:")


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
X_test = pd.DataFrame([[gender, age, loyal, travel_type, class_, distance]])

# предсказываем
col_1, col_2, col_3, col_4 = st.columns([0.6,0.01,0.16,0.35])

img_col, interp_col = st.columns([0.6,1])

with col_3:
    go = st.button('Предсказать')

result = 'dont_know'
if go:
    predict = load_model_and_predict(X_test, path='./data/little_model_weights.mv')
    if predict == 'satisfied':
        result = 'success'
    else:
        result = 'unsuccess'

with col_1:
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





with st.sidebar:
    with st.expander("Информация о предсказании"):
        st.write("Для предсказания была взята модель logistic regression. Основная причина - возможость предсказывать "
                 "экстраполированные данные. Особенно с учетом того, что практически все преобразования с исходным датасетом "
                 "приводили к жестким границам значений параметров. "
                 "Модель была обучена c гиперпараметрами по умолчанию (так как основная цель буткемпа- ML-сервис "
                 "(и я уже не успевал ничего))")