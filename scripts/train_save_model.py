import sys
from recosys.models.serialize import store, csv_path
from recosys.models.data_preparation import (
    read_uc_csv,
    transform_train_data,
    train_test_split,
    get_courses_names_dict,
)

from recosys.models.train_model import ItemBased


def main():
    df = read_uc_csv(csv_path("user_course"))
    df = transform_train_data(df)
    data_train, data_test = train_test_split(df)
    train_model = ItemBased(data_train)
    store_to_file(train_model, "item-based-model-train")
    store_to_file(data_train, "data-train")
    store_to_file(data_test, "data-test")
    store(get_courses_names_dict(csv_path("course")), "course_names")

    data_train, data_test = train_test_split(df, ratio=0.0)
    print(f"data_train.shape = {data_train.shape}, data_test.shape = {data_test.shape}")
    model = ItemBased(data_train)
    store_to_file(model, "item-based-model")


def store_to_file(data, filename="item-based-model"):
    store(data, filename)


if __name__ == "__main__":
    main()
    # try:
    #     main()
    # except Exception as e:
    #     sys.exit(1)
