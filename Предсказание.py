import pandas as pd
import streamlit as st
from ML import load_model_and_predict

st.title('Анализ данных о полете пассажиров.')

st.write("##### Предсказание, останется ли доволен пассажир полетом"
         "на основании основных характеристик пассажира.")

col1, col2, col3 = st.columns(3)
with col1:
    gender = st.radio('Выберите пол пассажира: ', ('Мужской', 'Женский'))
    loyal = st.radio('Лоялен ли клиент авиакомпании: ', ('Клиент лоялен', 'Клиент не лоялен'))
with col2:
    travel_type = st.radio('Выберите тип поездки: ', ('Деловая', 'Персональная'))
    class_ = st.radio('Выберите класс обслуживания: ', ('Бизнес', 'Эконом+', 'Эконом'))


with col3:
    age = st.number_input("Введите возраст пассажира (число полных лет)", step=1, value=25)
    distance = st.number_input("Укажите расстояние перелета (км)", value=2000)

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
distance = distance * 0.621371      # перевод в мили

# создаем X_test
X_test = pd.DataFrame([[gender, age, loyal, travel_type, class_, distance]])

# предсказываем
go = st.button('Предсказать удовлетворенность пассажира')
if go:
    predict = load_model_and_predict(X_test, path='./data/little_model_weights.mv')

    if predict == 'satisfied': st.write('Пассажир останется довольным!')
    else: st.write("К сожалению, пассажир не оценит перелет положительно.")

with st.expander("Примечание"):
    st.write("Для предсказания была взята модель logistic regression. Основная причина - возможость предсказывать "
             "экстраполированные данные. Особенно с учетом того, что практически все преобразования с исходным датасетом "
             "приводили к жестким границам значений параметров. "
             "Модель была обучена c гиперпараметрами по умолчанию (так как основная цель буткемпа- ML-сервис "
             "(и я уже не успевал ничего))")