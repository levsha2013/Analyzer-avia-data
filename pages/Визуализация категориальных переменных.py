import streamlit as st
from ML import *


st.title('Анализ данных о полете пассажиров.')
st.write("##### Визуализация категориальных данных.")
st.write("Рассматриваются только категориальные параметры пассажира: ")
st.write("*Пол, лояльность, тип поездки, класс обслуживания.*")

df = load_ds('./data/pilot.csv')
points = get_columns_to_preproc('bin')

with st.sidebar:
    point = st.radio('Параметр для анализа', ['Пол', 'Лояльность', "Тип поездки", "Класс обслуживания"])

dict_tmp = {
    'Пол': 'gender',
    'Лояльность': 'customer_type',
    'Тип поездки': 'type_of_travel',
    'Класс обслуживания': 'class'
}
fig = get_plot_category(df_x=df, column_x=dict_tmp[point], target_x='satisfaction')
st.pyplot(fig)

with st.expander("Примечание"):
    st.write("""Графики не корректно сравнивать в таком виде, так как есть небольшой дисбаланс классов в target
    переменной(43%/57%). Тем не менее основные тенденции влияния признаков на целевую переменную
    можно наблюдать без нормировки.""")