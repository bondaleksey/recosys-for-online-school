"""FastAPI course recommendation model"""
import pandas as pd
from fastapi import FastAPI, HTTPException

# from starlette_exporter import PrometheusMiddleware, handle_metrics
# from prometheus_client import Counter
from pydantic import BaseModel
from recosys.models.serialize import load


app = FastAPI()
# app.add_middleware(PrometheusMiddleware)
# app.add_route("/metrics", handle_metrics)
DATA = {}


class UserCourseLine(BaseModel):
    user_id: int
    course_id: int


class User(BaseModel):
    user_id: int


@app.on_event("startup")
def load_model():
    print("start_loading")
    DATA["train_model"] = load("item-based-model-train")
    DATA["model"] = load("item-based-model")
    DATA["data_train"] = load("data-train")
    DATA["data_test"] = load("data-test")
    DATA["course_names"] = load("course_names")
    # model = {"asd": 123}
    print("end_loading")


@app.get("/")
def read_healthcheck():
    return {"status": "Green", "version": "0.1.0"}


@app.post("/predict_uc_train")
def predict_uc_train(user_course: UserCourseLine):
    data = pd.DataFrame([user_course.dict()])
    try:
        pred = DATA["train_model"].predict(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"predict": str(pred[0])}


@app.post("/top_5_for_user_train")
def top_5_for_user_train(user_id=14312):
    user_id = int(user_id)
    predict = DATA["train_model"].top5_for_user(user_id)
    # print(f"predict = {predict}")
    if not isinstance(predict, str):
        prediction = {DATA["course_names"][k]: f" has rating {v}" for k, v in predict}
    else:
        prediction = predict

    res2 = [
        DATA["course_names"][course]
        for course in DATA["data_train"].query(f"user_id=={user_id}").course_id
    ]
    res3 = [
        DATA["course_names"][course]
        for course in DATA["data_test"].query(f"user_id=={user_id}").course_id
    ]
    return {
        "рекомендация": prediction,
        "модель знала, что пользователь интересовался": res2,
        "позже пользователь проявлял интерес к курсам": res3,
    }


@app.post("/predict_uc")
def predict_uc(user_course: UserCourseLine):
    data = pd.DataFrame([user_course.dict()])
    try:
        pred = DATA["model"].predict(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"predict": str(pred[0])}


@app.post("/top_5_for_user")
def top_5_for_user(user_id=14312):
    user_id = int(user_id)
    predict = DATA["model"].top5_for_user(user_id)
    print(f"predict = {predict}")
    if not isinstance(predict, str):
        prediction = {DATA["course_names"][k]: f" has rating {v}" for k, v in predict}
    else:
        prediction = predict

    res2 = [
        DATA["course_names"][course]
        for course in list(DATA["data_train"].query(f"user_id=={user_id}").course_id)
        + list(DATA["data_test"].query(f"user_id=={user_id}").course_id)
    ]

    return {
        "рекомендация": prediction,
        "модель знала, что пользователь интересовался": res2,
    }
