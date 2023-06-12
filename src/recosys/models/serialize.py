import os
import joblib
import pandas as pd

__all__ = ["store", "load", "csv_path"]


def store(model, filename, path="default"):
    if path == "default":
        path = models_path()
    filepath = os.path.join(path, filename + ".joblib")
    joblib.dump(model, filepath)


def load(filename, path="default"):
    if path == "default":
        path = models_path()
    filepath = os.path.join(path, filename + ".joblib")

    return joblib.load(filepath)


def csv_path(filename, path="default"):
    if path == "default":
        path = data_csv_path()
    filepath = os.path.join(path, filename + ".csv")
    return filepath


def models_path() -> str:
    script_path = os.path.abspath(__file__)
    script_dir_path = os.path.dirname(script_path)
    models_folder = os.path.join(script_dir_path, "..", "..", "..", "models")
    return models_folder


def data_csv_path() -> str:
    script_path = os.path.abspath(__file__)
    script_dir_path = os.path.dirname(script_path)
    models_folder = os.path.join(script_dir_path, "..", "..", "..", "data")
    return models_folder
