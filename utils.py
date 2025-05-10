import requests
import time
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from typing import Literal


def download_file(url: str):
    response = requests.get(url)
    response.raise_for_status()
    filename = url.split("/")[-1]
    with open(filename, "w") as f:
        f.write(response.text)
    return filename


def preprocess_pipeline(X: pd.DataFrame) -> tuple:
    cat = list(X.columns[X.dtypes == "object"])
    con = list(X.columns[X.dtypes != "object"])
    num_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    cat_pipe = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"),
    )
    pre = ColumnTransformer(
        [
            ("num", num_pipe, con),
            ("cat", cat_pipe, cat),
        ]
    )
    X_pre = pre.fit_transform(X)
    return X_pre, pre


def get_models() -> list:
    return [
        LogisticRegression(),
        DecisionTreeClassifier(),
        # RandomForestClassifier(),
        # GradientBoostingClassifier(),
        HistGradientBoostingClassifier(),
        XGBClassifier(),
    ]


def evaluate_model(model, xtrain, ytrain, xtest, ytest, n_jobs=-1):
    start = time.perf_counter()
    model.fit(xtrain, ytrain)
    ypred_train = model.predict(xtrain)
    ypred_test = model.predict(xtest)
    train_score = f1_score(ytrain, ypred_train, average="macro")
    test_score = f1_score(ytest, ypred_test, average="macro")
    scores = cross_val_score(
        model, xtrain, ytrain, cv=5, scoring="f1_macro", n_jobs=n_jobs
    )
    stop = time.perf_counter()
    elapsed = round(stop - start, 4)
    name = type(model).__name__
    res = {
        "name": name,
        "model": model,
        "train_score": round(train_score, 4),
        "test_score": round(test_score, 4),
        "cv_score": scores.mean().round(4),
        "model_time": elapsed,
    }
    print(res)
    return res


def evaluate_multiple(models: list, xtrain, ytrain, xtest, ytest):
    start = time.perf_counter()
    res = []
    for model in models:
        r = evaluate_model(model, xtrain, ytrain, xtest, ytest)
        res.append(r)
    res_df = pd.DataFrame(res)
    res_sorted = res_df.sort_values(by="cv_score", ascending=False).reset_index(
        drop=True
    )
    best_model = res_sorted.loc[0, "model"]
    stop = time.perf_counter()
    elapsed = round(stop - start, 4)
    print(f"Total time: {elapsed} seconds")
    return best_model, res_sorted


def evaluate_parallel(
    models: list, backend: Literal["threading", "loky"], xtrain, ytrain, xtest, ytest
):
    start = time.perf_counter()
    with parallel_backend(backend, n_jobs=-1):
        results = Parallel()(
            delayed(evaluate_model)(model, xtrain, ytrain, xtest, ytest, n_jobs=1)
            for model in models
        )
    res_df = pd.DataFrame(results)
    res_sorted = res_df.sort_values(by="cv_score", ascending=False).reset_index(
        drop=True
    )
    best_model = res_sorted.loc[0, "model"]
    stop = time.perf_counter()
    print(f"Total time: {round(stop - start, 4)} seconds")
    return best_model, res_sorted
