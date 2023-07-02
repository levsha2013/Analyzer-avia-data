import streamlit as st
from ML import *

df = load_ds('./data/pilot.csv')

st.title('Анализ данных о полете пассажиров.')
points = get_columns_to_preproc('5_point')


with st.sidebar:

    float_cols = get_columns_to_preproc('float')

    with st.sidebar:
        dict_name_cols = {
            'Возраст пассажира': 'age',
            'Расстояние перелета': 'flight_distance',
        }
        point = st.radio('Параметр для анализа', dict_name_cols.keys())
        status_2 = st.radio('Тип графика для целочисленных: ', ('countplot', 'barplot', 'histplot'))



tmp_dict = {
    'Нет': 'None',
    'Обрезать по границам': 'cut',
    'Преобразовать': 'prep'
}

point = dict_name_cols[point]
fig = get_plot_float(df_x=df, column_x=point, target_x='satisfaction', type_=status_2)
st.pyplot(fig)

with st.expander("Примечание"):
    st.write("Выше педставлены 3 графика: до обработки признака, при обработке обрезанием по порогу и при обработке пересчетом."
             "Порог обнаруживался эмпирически: визуально был четко выраженные спад по графикам countplot. "
             "Преобразование исходит из предположения о неверно введенных данных."
             "Таким образом значение признака делилось на 1000 (будто вводили не в км, а в метрах)")