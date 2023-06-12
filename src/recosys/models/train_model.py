import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# model = train_model(data)


class ItemBased:
    def __init__(self, data, user_col="user_id", item_col="course_id"):
        # data = X, y
        data = data.copy()
        # сохраним текущих пользователей и имеющиеся предметы
        self.users = data[user_col].unique()
        self.items = data[item_col].unique()

        # рассчитаем среднее значение рейтинга для пользователя и предмета
        self.mean_y_user = data.groupby(user_col)["rating"].mean()
        self.mean_y_item = data.groupby(item_col)["rating"].mean()

        # вычитаем среднюю оценку предмета
        # data["rating"] -= data[item_col].apply(lambda x: self.mean_y_item[x])

        # создаём векторы для каждого предмета с оценками пользователя
        # если пользователь не поставил оценку, то ставим 0
        self.item_ratings = pd.pivot_table(
            data, values="rating", index=item_col, columns=user_col, fill_value=0
        )

        # считаем попарную схожесть между фильмами
        self.item_sim = cosine_similarity(self.item_ratings)

        # также сделаем словарь {значение item_col: index в item_ratings}
        self.item_pos = dict()
        for item in self.items:
            self.item_pos[item] = np.argwhere(self.item_ratings.index.values == item)[
                0
            ][0]

    def predict_rating(self, pr_user, pr_item):
        # если в обучающей выборке нет такого предмета
        # или пользователя, то вернём 0
        if not pr_item in self.items or not pr_user in self.users:
            return 0

        # считаем числитель и знаменатель дроби из формулы предсказания
        numerator = self.item_sim[self.item_pos[pr_item]].dot(
            self.item_ratings.loc[:, pr_user]
        )
        if numerator == 0:
            return 0
        # вычитаем 1, так как схожесть предмета с самим собой равна 1,
        # но модель не должна это учитывать
        denominator = np.abs(self.item_sim[self.item_pos[pr_item]]).sum() - 1

        # return self.mean_y_item[pr_item] + numerator / denominator
        return round(numerator / denominator, 3)

    def predict_rating_avg(self, pr_user, pr_item):
        # если в обучающей выборке нет такого пользователя, то вернём 0
        # но если есть пользователь, предмет, а пользов, то вернем
        if not pr_item in self.items or not pr_user in self.users:
            if pr_user in self.users:
                return self.mean_y_user[pr_user]
            elif pr_item in self.items:
                return self.mean_y_item[pr_item]
            else:
                return 0

        # считаем числитель и знаменатель дроби из формулы предсказания
        numerator = self.item_sim[self.item_pos[pr_item]].dot(
            self.item_ratings.loc[:, pr_user]
        )
        if numerator == 0:
            return 0
        # вычитаем 1, так как схожесть предмета с самим собой равна 1,
        # но модель не должна это учитывать
        denominator = np.abs(self.item_sim[self.item_pos[pr_item]]).sum() - 1

        # return self.mean_y_item[pr_item] + numerator / denominator
        return round(numerator / denominator, 3)

    def predict(self, X_test, user_col="user_id", item_col="course_id"):
        y = X_test[[user_col, item_col]].apply(
            lambda row: self.predict_rating(row[0], row[1]), axis=1
        )
        return y

    def predict_avg(self, X_test, user_col="user_id", item_col="course_id"):
        y = X_test[[user_col, item_col]].apply(
            lambda row: self.predict_rating_avg(row[0], row[1]), axis=1
        )
        return y

    # надо еще функцию возвращающую лучших 5 курсов rating которых выше threshold
    def top5_for_user(self, pr_user, threshold=1.1):
        if not pr_user in self.item_ratings.columns:
            return "Нет пользователя c таким id"
        user_rating = self.item_ratings.loc[:, pr_user]
        interested_courses_id = [
            user_rating.index[item].values[0] for item in np.argwhere(user_rating > 0)
        ]

        res = [
            (item, round(self.predict_rating(pr_user, item), 2)) for item in self.items
        ]
        res = sorted(res, key=lambda x: x[1], reverse=True)[:25]

        res = [
            item
            for item in res
            if ((item[0] not in interested_courses_id) and item[1] > threshold)
        ]

        res = res[:5] if len(res) > 0 else "Нет хороших рекомендаций"
        return res

        # res = [(item, self.predict_rating_avg(pr_user, item)) for item in self.items]
        # res = sorted(res, key=lambda x: x[1], reverse=True)[:5]
        # print(type(res))
        # print(res)
        # return res


# class UserBased:
#     def __init__(self, data, user_col="user_id", item_col="course_id"):
#         X = X.copy()
#         # сохраним текущих пользователей и имеющиеся предметы
#         self.users = data[user_col].unique()
#         self.items = data[item_col].unique()

#         # рассчитаем среднее значение рейтинга для пользователя и предмета
#         self.mean_y_user = data.groupby(user_col)["rating"].mean()
#         self.mean_y_item = data.groupby(item_col)["rating"].mean()

#         # вычитаем среднюю оценку пользователя
#         data["rating"] -= data[user_col].apply(lambda x: self.mean_y_user[x])

#         # создаём векторы для каждого пользователя из набора использованных товаров
#         # для неизвестных товаров ставим оценку 0
#         self.user_ratings = pd.pivot_table(
#             data, values="rating", index=user_col, columns=item_col, fill_value=0
#         )

#         # считаем попарную схожесть между пользователями
#         self.user_sim = cosine_similarity(self.user_ratings)

#         # также сделаем словарь - {значение user_col: index в user_ratings = 1 значение (не вектор)}
#         self.user_pos = dict()
#         for user in self.users:
#             self.user_pos[user] = np.argwhere(self.user_ratings.index.values == user)[
#                 0
#             ][0]

#     def predict_rating(self, pr_user, pr_item):
#         # если в обучающей выборке нет такого предмета
#         # или пользователя, то вернём 0
#         if not pr_item in self.items or not pr_user in self.users:
#             return 0

#         # считаем числитель и знаменатель дроби из формулы предсказания
#         numerator = self.user_sim[self.user_pos[pr_user]].dot(
#             self.user_ratings.loc[:, pr_item]
#         )
#         # вычитаем 1, так как схожесть пользователя с самим собой равна 1,
#         # но модель не должна это учитывать
#         denominator = np.abs(self.user_sim[self.user_pos[pr_user]]).sum() - 1

#         return self.mean_y_user[pr_user] + numerator / denominator

#     def predict_rating_avg(self, pr_user, pr_item):
#         # если в обучающей выборке нет такого предмета
#         # или пользователя, то вернём 0
#         if not pr_item in self.items or not pr_user in self.users:
#             if pr_user in self.users:
#                 return self.mean_y_user[pr_user]
#             elif pr_item in self.items:
#                 return self.mean_y_item[pr_item]
#             else:
#                 return 0

#         # считаем числитель и знаменатель дроби из формулы предсказания
#         numerator = self.user_sim[self.user_pos[pr_user]].dot(
#             self.user_ratings.loc[:, pr_item]
#         )
#         # вычитаем 1, так как схожесть пользователя с самим собой равна 1,
#         # но модель не должна это учитывать
#         denominator = np.abs(self.user_sim[self.user_pos[pr_user]]).sum() - 1

#         return self.mean_y_user[pr_user] + numerator / denominator

#     def predict(self, X, user_col="user_id", item_col="course_id"):
#         y = X[[user_col, item_col]].apply(
#             lambda row: self.predict_rating(row[0], row[1]), axis=1
#         )
#         return y

#     def predict_avg(self, X, user_col="user_id", item_col="course_id"):
#         y = X[[user_col, item_col]].apply(
#             lambda row: self.predict_rating_avg(row[0], row[1]), axis=1
#         )
#         return y
