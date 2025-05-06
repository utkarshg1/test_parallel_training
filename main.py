from utils import (
    download_file,
    preprocess_pipeline,
    get_models,
    evaluate_multiple,
    evaluate_parallel,
)
import pandas as pd
from sklearn.model_selection import train_test_split

URL = "https://raw.githubusercontent.com/utkarshg1/Machine-Learning-Ravet-5-7pm/refs/heads/main/15%20Classification%20Project/train_machine.csv"

FILE_NAME = download_file(URL)


def main():
    df = pd.read_csv(FILE_NAME)
    X = df.drop(columns=["id", "Product ID", "Machine failure"])
    y = df[["Machine failure"]].values.flatten()
    X_pre, pre = preprocess_pipeline(X)
    xtrain, xtest, ytrain, ytest = train_test_split(
        X_pre, y, test_size=0.2, random_state=42
    )
    models = get_models()
    # Evaluate models sequentially
    best_model1, res1 = evaluate_multiple(models, xtrain, ytrain, xtest, ytest)
    # Evaluate models in parallel
    best_model2, res2 = evaluate_parallel(models, xtrain, ytrain, xtest, ytest)
    # Display results
    print("Sequential Evaluation Results:")
    print(res1)
    print("Parallel Evaluation Results:")
    print(res2)


if __name__ == "__main__":
    main()
