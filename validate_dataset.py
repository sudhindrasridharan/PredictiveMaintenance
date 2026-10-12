
import os
import glob
import sys
import pandas as pd

# Project paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# Expected columns from the project document
EXPECTED_COLUMNS = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp",
    "Engine Condition"
]

def validate_dataset():

    print("=" * 60)
    print("PREDICTIVE MAINTENANCE - DATA REGISTRATION")
    print("=" * 60)

    # Locate CSV files in the data folder
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

    if not csv_files:
        print("ERROR: No CSV dataset found in the data folder.")
        print("Please upload the dataset CSV and run the script again.")
        sys.exit(1)

    if len(csv_files) > 1:
        print("ERROR: Multiple CSV files found:")
        for file in csv_files:
            print("-", os.path.basename(file))
        print("Keep only the intended dataset CSV in the data folder.")
        sys.exit(1)

    csv_path = csv_files[0]

    print("\nDataset file:", os.path.basename(csv_path))
    print("Dataset path:", csv_path)

    # Load dataset
    try:
        df = pd.read_csv(csv_path)
    except Exception as error:
        print("ERROR: Unable to read the CSV file:", error)
        sys.exit(1)

    # Validate expected columns
    missing_columns = [
        col for col in EXPECTED_COLUMNS if col not in df.columns
    ]

    unexpected_columns = [
        col for col in df.columns if col not in EXPECTED_COLUMNS
    ]

    print("\n--- Column Validation ---")
    if missing_columns:
        print("VALIDATION FAILED: Expected columns are missing.")
        print("Missing columns:", missing_columns)
    else:
        print("VALIDATION PASSED: All expected columns are present.")

    if unexpected_columns:
        print("Additional columns found:", unexpected_columns)
    else:
        print("No additional columns found.")

    print("\nExpected column count:", len(EXPECTED_COLUMNS))
    print("Actual column count:", len(df.columns))

    # Dataset summary
    print("\n--- Dataset Summary ---")
    print("Number of rows:", df.shape[0])
    print("Number of columns:", df.shape[1])
    print("Dataset dimensions:", df.shape)

    print("\n--- Column Names ---")
    print(df.columns.tolist())

    print("\n--- Data Types ---")
    print(df.dtypes)

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    print("\nTotal missing values:", int(df.isnull().sum().sum()))
    print("Duplicate rows:", int(df.duplicated().sum()))

    # Target variable summary, if available
    if "Engine_Condition" in df.columns:
        print("\n--- Engine Condition Distribution ---")
        print(df["Engine_Condition"].value_counts(dropna=False))

    print("\n--- Numerical Summary ---")
    print(df.describe().to_string())

    # Final validation status
    print("\n--- Final Registration Status ---")

    if missing_columns:
        print("DATA REGISTRATION FAILED: Please review the missing columns.")
        sys.exit(1)

    print("DATA REGISTRATION SUCCESSFUL")
    print("The dataset passed the expected-column validation.")


if __name__ == "__main__":
    validate_dataset()
