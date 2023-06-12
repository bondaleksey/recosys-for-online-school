import pandas as pd
import numpy as np
from datetime import datetime

# from sklearn.metrics.pairwise import cosine_similarity

# data = data_transformation(read_uc_csv(csv_path("user_course")))


# def my_ts_converter(val):
#     if len(val) in (10, 13):
#         try:
#             val = int(val)
#             if val > 10**11:
#                 val = int(val / 1000)
#             return datetime.fromtimestamp(val)
#         except:
#             return np.NaN
#     return np.NaN

__all__ = ["read_uc_csv", "transform_train_data", "train_test_split"]


def my_int_converter(val):
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, str):
        try:
            return float(val)
        except:
            return np.NaN
    return np.NaN


def my_date_converter(val):
    try:
        return datetime.strptime(val, "%d.%m.%Y %H:%M")
    except:
        return np.NaN


def read_uc_csv(filepath):
    def get_rating(row):
        if row["certificate_hash"]:
            return 10
        elif row["group_id"] > 0:
            return 6
        elif row["assessment_id"] > 0:
            return 3
        else:
            return 1

    # прочитать, только нужные столбцы из csv
    ucols = [
        "user_id",
        "course_id",
        "created",
        "certificate_hash",
        "group_id",
        "assessment_id",
    ]
    date_cols = ["created"]
    int_cols = ["user_id", "course_id", "group_id", "assessment_id"]
    str_cols = ["certificate_hash"]
    converters_uc = {item: my_int_converter for item in int_cols}
    converters_uc.update({item: my_date_converter for item in date_cols})
    converters_uc.update({item: str for item in str_cols})

    data = pd.read_csv(filepath, usecols=ucols, sep=";", converters=converters_uc)
    # "../data/user_course.csv", usecols=ucols, sep=";", converters=converters_uc
    data.sort_values(by="created")
    # условия для рейтинга:
    # есть диплом -> r = 1 (df_user_course['certificate_hash'].notna())
    # оплатил курс -> r += 0.4 (Хотя бы для одной из записей full_paid == 1 or group_id not None)
    # сдан вступительный тест -> r += 0.3 (Хотя бы для одной из записей assessment_id not NaN )
    # есть только запись в таблице, а больше ничего нет то ставим 0.1
    # Условия на 0 <= r <= 1

    data["rating"] = data.apply(get_rating, axis=1)
    data = data.drop(labels=["group_id", "certificate_hash", "assessment_id"], axis=1)
    data = data.dropna()
    data[["user_id", "course_id"]] = data[["user_id", "course_id"]].astype(int)

    # В data остались повторные записи, их надо почистить!
    return data


def drop_duplicates(df, ids, columns, ascending):
    """Function delete duplicates in cols and leave min (ascending==1) or max (ascending==0) for unique values in ids

    Args:
        df (DataFrame): initial dataframe
        ids (list): columns names of user_id and product_id
        columns (list): list of columns names
        ascending (list): list of bools values associated with columns

    Returns:
        DataFrame:
    """
    cols_iterator = enumerate(columns)
    i, col = next(cols_iterator)
    ndf = (
        df.sort_values(col, ascending=ascending[i])
        .drop_duplicates(ids)
        .set_index(ids)[[col]]
    )
    for i, col in cols_iterator:
        ndf[col] = (
            df.sort_values(col, ascending=ascending[i])
            .drop_duplicates(ids)
            .set_index(ids)[[col]]
        )
    return ndf.reset_index()


# df = drop_duplicates(
#     user_course_table(), ["user_id", "course_id"], ["date", "rating"], [1, 0]
# )


def transform_train_data(uc_data):
    return drop_duplicates(
        uc_data, ["user_id", "course_id"], ["created", "rating"], [1, 0]
    )


def train_test_split(
    X,
    ratio=0.3,
    user_col="user_id",
    item_col="course_id",
    rating_col="rating",
    time_col="created",
):
    # сортируем оценки по времени
    X.sort_values(by=[time_col], inplace=True)
    # список всех юзеров
    userIds = X[user_col].unique()
    X_train_data = []
    X_test_data = []
    for userId in userIds:
        curUser = X[X[user_col] == userId]
        # определяем позицию, по которой делим выборку и размещаем данные по массивам
        if curUser.shape[0] <= 1 and ratio > 0.01:
            continue
        idx = int(curUser.shape[0] * (1 - ratio))
        X_train_data.append(
            curUser[[user_col, item_col, rating_col]].iloc[:idx, :].values
        )
        X_test_data.append(
            curUser[[user_col, item_col, rating_col]].iloc[idx:, :].values
        )
    # cтекуем данные по каждому пользователю в общие массивы
    data_train = pd.DataFrame(
        np.vstack(X_train_data), columns=[user_col, item_col, rating_col]
    )
    data_test = pd.DataFrame(
        np.vstack(X_test_data), columns=[user_col, item_col, rating_col]
    )
    return data_train, data_test


def get_courses_names_dict(filepath: str) -> dict:
    ucols = ["id", "title"]
    converters_uc = {"id": my_int_converter, "title": str}
    df_course = pd.read_csv(filepath, usecols=ucols, sep=";", converters=converters_uc)
    df_course.dropna()
    df_course = df_course.astype({"id": "int"})
    return df_course.set_index("id").to_dict()["title"]
