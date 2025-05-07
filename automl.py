import time
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier


class BaseAutoML:
    def __init__(
        self,
        data: pd.DataFrame,
        drop_columns: list[str],
        target_column: str,
        test_size: float = 0.2,
    ) -> None:
        """
        Initialize the BaseAutoML class.

        Parameters:
        - data (pd.DataFrame): The input data.
        - drop_columns (list[str]): List of columns to drop from the data.
        - target_column (str): The target column name.
        """
        self.data = data.copy()
        self.drop_columns = drop_columns
        self.target_column = target_column
        self.X = None
        self.y = None
        self.test_size = test_size
        self.xtrain = None
        self.xtest = None
        self.ytrain = None
        self.ytest = None
        self.preprocessor = None
        self.best_model = None
        self.results = None

    def preprocess_data(self) -> None:
        """
        Preprocess the data by dropping specified columns and applying the preprocessing pipeline.
        """
        self.data.drop(columns=self.drop_columns, inplace=True)
        self.X = self.data.drop(columns=[self.target_column])
        self.y = self.data[self.target_column].values.flatten()

        cat = list(self.X.columns[self.X.dtypes == "object"])
        con = list(self.X.columns[self.X.dtypes != "object"])

        num_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

        if cat:
            cat_pipe = make_pipeline(
                SimpleImputer(strategy="most_frequent"), OneHotEncoder()
            )
            self.preprocessor = ColumnTransformer(
                transformers=[("num", num_pipe, con), ("cat", cat_pipe, cat)]
            )
        else:
            self.preprocessor = ColumnTransformer(transformers=[("num", num_pipe, con)])

        self.X = self.preprocessor.fit_transform(self.X)

        return self.X, self.y, self.preprocessor

    def split_data(self) -> tuple:
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=42
        )
        return self.xtrain, self.xtest, self.ytrain, self.ytest

    def evaluate_model(self, model, xtrain, ytrain, scoring) -> dict:
        start = time.perf_counter()
        model.fit(xtrain, ytrain)
        scores = cross_val_score(
            model, xtrain, ytrain, cv=5, scoring=scoring, n_jobs=-1
        )
        stop = time.perf_counter()
        name = model.__class__.__name__
        elapsed_time = stop - start
        mean_score = scores.mean()
        print(
            f"Model: {name}, Time taken: {elapsed_time:.2f} seconds, Score: {mean_score:.4f}"
        )
        return {
            "name": name,
            "model": model,
            "score": mean_score,
            "time": elapsed_time,
        }

    def evaluate_multiple(self, models: list, xtrain, ytrain, scoring) -> tuple:
        """
        Evaluate multiple models in parallel and return the best one.

        Parameters:
        - models (list): List of models to evaluate
        - xtrain: Training data features
        - ytrain: Training data target
        - scoring: Scoring metric for cross-validation

        Returns:
        - tuple: Best model and results DataFrame
        """
        from joblib import Parallel, delayed

        print("Evaluating models in parallel...")
        results = Parallel(n_jobs=-1)(
            delayed(self.evaluate_model)(model, xtrain, ytrain, scoring)
            for model in models
        )

        results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
        self.results = pd.DataFrame(results_sorted)
        self.best_model = self.results.loc[0]["model"]

        print(
            f"Best model: {self.best_model.__class__.__name__} with score: {self.results.loc[0]['score']:.4f}"
        )
        return self.best_model, self.results


class RegressionAutoML(BaseAutoML):
    def __init__(
        self, data: pd.DataFrame, drop_columns: list[str], target_column: str
    ) -> None:
        super().__init__(data, drop_columns, target_column)

    def get_models(self) -> list:
        return [
            LinearRegression(),
            Ridge(),
            Lasso(),
            RandomForestRegressor(),
            XGBRegressor(eval_metric="rmse"),
        ]

    def run(self) -> tuple:
        """
        Run the full AutoML pipeline for regression tasks.

        Returns:
        - tuple: Best model and results DataFrame
        """
        print("Starting AutoML for regression...")
        self.preprocess_data()
        self.split_data()
        models = self.get_models()

        # Use R2 score as default metric for regression
        best_model, results = self.evaluate_multiple(
            models, self.xtrain, self.ytrain, scoring="r2"
        )

        # Evaluate best model on test set
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        import numpy as np

        y_pred = best_model.predict(self.xtest)
        r2 = r2_score(self.ytest, y_pred)
        rmse = np.sqrt(mean_squared_error(self.ytest, y_pred))
        mae = mean_absolute_error(self.ytest, y_pred)

        print(f"\nBest model performance on test set:")
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")

        return best_model, results


class ClassificationAutoML(BaseAutoML):
    def __init__(
        self,
        data: pd.DataFrame,
        drop_columns: list[str],
        target_column: str,
        test_size: float = 0.2,
        multi_class: bool = False,
    ) -> None:
        super().__init__(data, drop_columns, target_column, test_size)
        self.multi_class = multi_class

    def get_models(self) -> list:
        """
        Get a list of classification models to evaluate.

        Returns:
        - list: List of classification model instances
        """

        return [
            LogisticRegression(max_iter=1000),
            DecisionTreeClassifier(),
            RandomForestClassifier(),
            XGBClassifier(
                use_label_encoder=False,
                eval_metric="mlogloss" if self.multi_class else "logloss",
            ),
        ]

    def run(self) -> tuple:
        """
        Run the full AutoML pipeline for classification tasks.

        Returns:
        - tuple: Best model and results DataFrame
        """
        print("Starting AutoML for classification...")
        self.preprocess_data()
        self.split_data()
        models = self.get_models()

        # Use appropriate scoring metric
        scoring = "f1_macro" if self.multi_class else "f1"
        best_model, results = self.evaluate_multiple(
            models, self.xtrain, self.ytrain, scoring=scoring
        )

        # Evaluate best model on test set
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
            classification_report,
        )

        y_pred = best_model.predict(self.xtest)
        accuracy = accuracy_score(self.ytest, y_pred)
        precision = precision_score(
            self.ytest, y_pred, average="macro" if self.multi_class else "binary"
        )
        recall = recall_score(
            self.ytest, y_pred, average="macro" if self.multi_class else "binary"
        )
        f1 = f1_score(
            self.ytest, y_pred, average="macro" if self.multi_class else "binary"
        )

        print(f"\nBest model performance on test set:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        print("\nConfusion Matrix:")
        print(confusion_matrix(self.ytest, y_pred))

        print("\nClassification Report:")
        print(classification_report(self.ytest, y_pred))

        return best_model, results
