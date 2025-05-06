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
from xgboost import XGBClassifier
import pandas as pd
from joblib import Parallel, delayed


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
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        HistGradientBoostingClassifier(),
        XGBClassifier(),
    ]


def evaluate_model(model, xtrain, ytrain, xtest, ytest, n_jobs=-1):
    start = time.perf_counter()
    model.fit(xtrain, ytrain)
    train_score = model.score(xtrain, ytrain)
    test_score = model.score(xtest, ytest)
    scores = cross_val_score(
        model, xtrain, ytrain, cv=5, scoring="f1_macro", n_jobs=n_jobs
    )
    stop = time.perf_counter()
    elapsed = round(stop - start, 4)
    name = type(model).__name__
    return {
        "name": name,
        "model": model,
        "train_score": train_score,
        "test_score": test_score,
        "cv_score": scores.mean(),
        "model_time": elapsed,
    }


def evaluate_multiple(models: list, xtrain, ytrain, xtest, ytest):
    start = time.perf_counter()
    res = []
    for model in models:
        r = evaluate_model(model, xtrain, ytrain, xtest, ytest)
        res.append(r)
        print(r)
    res_df = pd.DataFrame(res)
    res_sorted = res_df.sort_values(by="cv_score", ascending=False).reset_index(
        drop=True
    )
    best_model = res_sorted.loc[0, "model"]
    stop = time.perf_counter()
    elapsed = round(stop - start, 4)
    print(f"Total time: {elapsed} seconds")
    return best_model, res_sorted


def evaluate_parallel(models: list, xtrain, ytrain, xtest, ytest):
    start = time.perf_counter()
    results = Parallel(n_jobs=-1)(
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
