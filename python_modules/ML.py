import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from pickle import dump, load


RANDOM_STATE = 42

DATASET_PATH = "https://raw.githubusercontent.com/evgpat/edu_stepik_from_idea_to_mvp/main/datasets/clients.csv"
# DATASET_PATH = 'pilot.csv'


# чтение, выделение столбцов, удаление na и "-"
def load_ds(path):
    """
    Принимает путь до df или URL-ссылку

    Загружает df
    Изменяет названия столбцов и строк
    Удаляет все объекты с nan
    Удаляет satisfaction == '-'

    Возвращает df"""
    df = pd.read_csv(path)
    df.columns = [i.replace(" ", "_").lower() for i in df.columns]
    df.dropna(inplace=True)
    df = df[df['satisfaction'] != '-']
    return df


def get_columns_to_preproc(type_='point'):
    """
    point - возвращает список столбцов, которые означают оценку. Стоит по умолчанию
    bin - возвращает список столбцов, которые можно бинаризовать или OHE с удалением столбца
    float - вохвращает список столбцов, которые являются целочисленными
    """
    if type_ == '5_point':
        return ['inflight_wifi_service', 'departure/arrival_time_convenient',
                'ease_of_online_booking', 'gate_location', 'food_and_drink',
                'online_boarding', 'seat_comfort', 'inflight_entertainment',
                'on-board_service', 'leg_room_service', 'baggage_handling',
                'checkin_service', 'inflight_service', 'cleanliness']

    elif type_ == 'bin':
        return ['gender', 'customer_type', 'type_of_travel', 'class']

    elif type_ == 'float':
        return ['age', 'flight_distance', 'departure_delay_in_minutes',
                'arrival_delay_in_minutes']


# Обработка
def to_5_points(df_x, column_x):
    """

    :param df_x: входной DataFrame
    :param column_x: список признаков
    :return: измененный DataFrame
    """
    df_x = df_x.copy()
    if type(column_x) == list:
        for col_x in column_x:
            df_x[col_x] = df_x[col_x].apply(lambda x: int(x) if x <= 5 else int(round(x / 10)))
    return df_x


def binariser(df_x, column_x):
    """
    :param df_x: входной DataFrame
    :param column_x: список признаков для преобразования
    :return: преобразованные DataFrame
    """
    for col_x in column_x:
        if col_x == 'gender':
            tmp_dict = {
                'Male': 1,
                'Female': 0}
            df_x.loc[:, col_x] = df_x.loc[:, col_x].apply(lambda x: tmp_dict[x])
        elif col_x == 'customer_type':
            tmp_dict = {
                'disloyal Customer': 0,
                'Loyal Customer': 1}
            df_x.loc[:, col_x] = df_x.loc[:, col_x].apply(lambda x: tmp_dict[x])
        elif col_x == 'type_of_travel':
            tmp_dict = {
                'Personal Travel': 0,
                'Business travel': 1}
            df_x.loc[:, col_x] = df_x.loc[:, col_x].apply(lambda x: tmp_dict[x])
        # это сделано специально для class
        # однако там 3 значения: экономия, экономия+ и бизнес.
        # можно перевести в ранговую шкалу
        elif col_x == 'class':
            tmp_dict = {
                'Eco': 0,
                'Eco Plus': 1,
                'Business': 2}
            df_x.loc[:, col_x] = df_x.loc[:, col_x].apply(lambda x: tmp_dict[x])
        elif col_x == 'satisfaction':
            tmp_dict = {
                'satisfied': 1,
                'neutral or dissatisfied': 0}
            df_x.loc[:, col_x] = df_x.loc[:, col_x].apply(lambda x: tmp_dict[x])

        # на всякий случай оставлю
        else:
            tmp = pd.get_dummies(df_x[column_x], drop_first=True)
            tmp.columns = [f"{column_x}_{i}" for i in tmp.columns]
            df_x = pd.concat([df_x.drop(column_x, axis=1), tmp], axis=1)
    return df_x


def float_prep(df_x, columns_x, type_='cut'):
    """
    :param df_x: Входной DataFrame
    :param columns_x: список признаков для преобразования
    :param type: 'cut' - обрезка / 'prep' - преобразование предумсотренное
    :return: Выходной DataFrame
    """
    # сразу обрезаем людей с возрастом 0 лет

    df_x = df_x[df_x['age'] > 0]

    for col_x in columns_x:
        if col_x == 'age' and type_ == 'cut':
            df_x = df_x[(df_x['age'] <= 70) & (df_x['age'] >= 7)]
        elif col_x == 'age' and type_ == 'prep':
            df_x.loc[:, 'age'] = df_x.loc[:, 'age'].apply(lambda x: x if x <= 70 else round(x // 10))

        if col_x == 'flight_distance' and type_ == 'cut':
            df_x = df_x[df_x['flight_distance'] < 4000]
        elif col_x == 'flight_distance' and type_ == 'prep':
            df_x.loc[:, 'age'] = df_x.loc[:, 'age'].apply(lambda x: x if x <= 4000 else round(x // 1000))

    """departure_delay_in_minutes и arrive delay не имею ярковыраженных границ.
    Однако разумные пределы они все равно превышают (задержка прибытия самолета - 10 дней)
    Логичнее всего будет их удалить."""
    if 'departure_delay_in_minutes' in df_x.columns:
        df_x.drop('departure_delay_in_minutes', axis=1, inplace=True)
    if 'arrival_delay_in_minutes' in df_x.columns:
        df_x.drop('arrival_delay_in_minutes', axis=1, inplace=True)

    return df_x


# Графики
def get_plot_points(df_x, column_x, target_x, type_='countplot'):
    """

    :param df_x: входной df
    :param column_x: интересующий признак (имя столбца):
    :param target_x: второй интересующий признак (имя столбца, обычно таргет):
    :param type_: == count bar hist box
    :return: фигуру, которую потом можно отрисовать
    """
    fig, ax = plt.subplots(figsize=(10,4))

    #  нужно представлять в виде процентов, а не абсолютными, нечестно немного получается
    if type_ == "countplot":
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        sns.countplot(x=column_x, hue=target_x, data=df_x)
        plt.subplot(2, 1, 2)
        sns.countplot(x=column_x, hue=target_x, data=to_5_points(df_x, [column_x]))

    elif type_ == "barplot":
        plt.subplot(1,2,1)
        sns.barplot(y=column_x, x=target_x, data=df_x)
        plt.subplot(1, 2, 2)
        sns.barplot(y=column_x, x=target_x, data=to_5_points(df_x, [column_x]))

    elif type_ == 'histplot':
        plt.subplot(1, 2, 1)
        sns.histplot(x=column_x, data=df_x, hue=target_x)
        plt.subplot(1, 2, 2)
        sns.histplot(x=column_x, hue=target_x, data=to_5_points(df_x, [column_x]))
    elif type_ == 'boxplot':
        plt.subplot(1,2,1)
        sns.boxplot(y=column_x, hue=target_x, data=df_x)
        plt.subplot(1, 2, 2)
        sns.boxplot(y=column_x, hue=target_x, data=to_5_points(df_x, [column_x]))
    # странно выводит график, мне не нравится, не использую
    # elif type_ == 'join':
    #   sns.jointplot(x=column_x, y=target_x, data=df_x);
    return fig


def get_plot_float(df_x, column_x, target_x, type_='countplot'):
    fig, ax = plt.subplots(figsize=(25,18))

    #  нужно представлять в виде процентов, а не абсолютными, нечестно немного получается
    if type_ == "countplot" and column_x == 'age':
        plt.subplot(3,1,1)
        g = sns.countplot(x=column_x, hue=target_x, data=df_x[df_x['age'] < 100])
        g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.subplot(3,1,2)
        sns.countplot(x=column_x, hue=target_x, data=float_prep(df_x, [column_x], type_='cut'))
        plt.subplot(3,1,3)
        sns.countplot(x=column_x, hue=target_x, data=float_prep(df_x, [column_x], type_='prep'))

    if type_ == "countplot" and column_x == 'flight_distance':
        plt.subplot(3,1,1)
        g = sns.countplot(x=column_x, hue=target_x, data=df_x[df_x['flight_distance'] < 5000])
        g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.subplot(3,1,2)
        sns.countplot(x=column_x, hue=target_x, data=float_prep(df_x, [column_x], type_='cut'))
        plt.subplot(3,1,3)
        sns.countplot(x=column_x, hue=target_x, data=float_prep(df_x, [column_x], type_='prep'))
    elif type_ == "barplot":
        sns.barplot(y=column_x, x=target_x, data=df_x)
    return fig


def get_plot_category(df_x, column_x, target_x):
    """

    :param df_x: входной df
    :param column_x: интересующий признак (имя столбца):
    :param target_x: второй интересующий признак (имя столбца, обычно таргет):
    :return: фигуру, которую потом можно отрисовать
    """

    fig, ax = plt.subplots(figsize=(13,7))
    plt.subplot(1,2,1)
    sns.countplot(x=column_x, hue=target_x, data=df_x)
    plt.subplot(1, 2, 2)
    sns.histplot(x=column_x, data=df_x, hue=target_x)

    return fig


# ML
def learn_model(df_x, path='./data/model_weights.mv'):

    X_train, X_test, y_train, y_test = train_test_split(df_x.drop('satisfaction', axis=1),
                                                        df_x['satisfaction'],
                                                        random_state=42,
                                                        train_size=0.8)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    score = model.score(X_test, y_test)

    with open(path, "wb") as file:
        dump(model, file)

    print(f'Качество обученной модели по ROC-AUC - {score}')


def learn_little_model(df_x, path='./data/little_model_weights.mv'):
    df_x = df_x[['gender', 'age', 'customer_type', 'type_of_travel', 'class', 'flight_distance', 'satisfaction']]
    X_train, X_test, y_train, y_test = train_test_split(df_x.drop('satisfaction', axis=1),
                                                        df_x['satisfaction'],
                                                        random_state=42,
                                                        train_size=0.8)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    score = model.score(X_test, y_test)

    with open(path, "wb") as file:
        dump(model, file)

    print(f'Качество обученной модели по ROC-AUC - {score}')


def load_model_and_predict(X_test_x, path='./data/model_weights.mv'):
    with open(path, "rb") as file:
        model = load(file)

    return model.predict(X_test_x)


if __name__ == '__main__':
    df_start = load_ds('./data/pilot.csv')
    y = 'satisfaction'

    points = get_columns_to_preproc('5_point')
    to_binarizer = get_columns_to_preproc('bin')
    other_preprocessing = get_columns_to_preproc('float')

    df_tmp = to_5_points(df_start, points)
    # df_tmp = binariser(df_tmp, to_binarizer)
    df_tmp = float_prep(df_tmp, other_preprocessing, type_='prep')

    # тут отрисовка всяких графиков

    # to_binarizer.append('satisfaction')
    df_tmp = binariser(df_tmp, to_binarizer)
    df_tmp = df_tmp.drop(['id'], axis=1)
    print(df_tmp.columns)

    learn_model(df_x=df_tmp)
    learn_little_model(df_x=df_tmp)


    print('Конец работы программы')
