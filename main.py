from utils import download_file
import pandas as pd
from automl import ClassificationAutoML, RegressionAutoML
import time

# URLs for the datasets
MACHINE_URL = "https://raw.githubusercontent.com/utkarshg1/Machine-Learning-Ravet-5-7pm/refs/heads/main/15%20Classification%20Project/train_machine.csv"
LOAN_URL = "train_loan.csv"  # Assuming this file is already in the workspace


def main():
    # Classification example with machine failure dataset
    print("=" * 50)
    print("Classification Example - Machine Failure Prediction")
    print("=" * 50)

    # Download the file if needed
    try:
        machine_df = pd.read_csv("train_machine.csv")
        print("Using existing train_machine.csv file")
    except FileNotFoundError:
        machine_file = download_file(MACHINE_URL)
        machine_df = pd.read_csv(machine_file)
        print(f"Downloaded and loaded {machine_file}")

    # Initialize the ClassificationAutoML
    machine_automl = ClassificationAutoML(
        data=machine_df,
        drop_columns=["id", "Product ID"],
        target_column="Machine failure",
        test_size=0.2,
    )

    # Run the classification pipeline
    start_time = time.perf_counter()
    best_model, results = machine_automl.run()
    end_time = time.perf_counter()
    print(f"Total Classification AutoML time: {end_time - start_time:.2f} seconds")
    print("\nTop 3 models:")
    print(results[["name", "score", "time"]].head(3))

    # Regression example with loan dataset (if available)
    try:
        print("\n" + "=" * 50)
        print("Regression Example - Loan Amount Prediction")
        print("=" * 50)

        loan_df = pd.read_csv(LOAN_URL)
        print(f"Loaded {LOAN_URL}")

        # Check if the loan dataset has a numeric target column for regression
        # Here we're assuming 'LoanAmount' is the target column
        if "LoanAmount" in loan_df.columns and pd.api.types.is_numeric_dtype(
            loan_df["LoanAmount"]
        ):
            # Initialize the RegressionAutoML
            loan_automl = RegressionAutoML(
                data=loan_df,
                drop_columns=["Loan_ID"],  # Adjust based on actual columns
                target_column="LoanAmount",
            )

            # Run the regression pipeline
            start_time = time.perf_counter()
            best_model, results = loan_automl.run()
            end_time = time.perf_counter()
            print(f"Total Regression AutoML time: {end_time - start_time:.2f} seconds")
            print("\nTop 3 models:")
            print(results[["name", "score", "time"]].head(3))
        else:
            print(
                f"Loan dataset does not have a suitable numeric target column for regression."
            )
    except FileNotFoundError:
        print(f"Loan dataset file '{LOAN_URL}' not found. Skipping regression example.")


if __name__ == "__main__":
    main()
