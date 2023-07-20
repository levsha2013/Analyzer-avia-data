from catboost import CatBoostClassifier
import pandas as pd


def get_interpretation(gender, age, loyal, travel_type, class_, distance):
    """

    :param gender: Пол 0 - Female, 1 - Male
    :param age: age of passager
    :param loyal: 0 - Disloyal, 1 - Loyal
    :param travel_type: 0 - Presonal travel, 1- Buisness travel
    :param class_: 0 - Eco, 1 - Eco+, 2 - Buisness
    :param distance: flight distance
    :return: [bool] have enought data or not to predict
    if True: [DataFrame] feature importance

    """
    have_data = 0
    # преобразование к нормальным величинам:
    main_columns = ['gender', 'customer_type', 'type_of_travel', 'class', 'age', 'flight_distance']
    df_x = pd.read_csv('./data/pilot_prep_factor.csv')
    print(max(df_x['flight_distance']))
    print(gender, age, loyal, travel_type, class_, distance)
    df_x = df_x[(df_x['gender'] == gender) &
                (df_x['customer_type'] == loyal) &
                (df_x['type_of_travel'] == travel_type) &
                (df_x['class'] == class_) &
                ((df_x['age'] >= (age - 5)) & (df_x['age'] <= age+5)) &
                ((df_x['flight_distance'] >= (distance - 250)) & (df_x['flight_distance'] <= distance + 250))].drop(main_columns, axis=1)
    if df_x.shape[0] > 1:
        print(df_x.shape[0])

        # данных для обработки достаточно, меняем флаг на Ture,
        have_data = True

        # выводим средние оценки
        check = df_x.drop('satisfaction', axis=1).mean()

        # смотрим среднее довольство таких людей (в процентах)
        mean_satisfaction = round(df_x['satisfaction'].mean()*100, 2)

        # и обучаем CatBoost для feature importance
        catboos = CatBoostClassifier(n_estimators=100, random_state=42)
        X = df_x.drop(['satisfaction'], axis=1)
        y = df_x[['satisfaction']]

        catboos.fit(X, y, verbose=0)
        result = pd.Series(catboos.feature_importances_, index=X.columns).sort_values(ascending=False)
        return have_data, result, check, mean_satisfaction
    else:
        print(df_x.shape[0])
        return have_data, None, None, None
