import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_cicids_dataset(path: str, sample_frac: float = 1.0, random_state: int = 42):
    """
    Load and preprocess the CICIDS-2018 intrusion detection dataset.

    This function performs the full preprocessing pipeline for network traffic data:
    1. Reads the dataset from a CSV file.
    2. Drops columns that contain only NaN values.
    3. Optionally subsamples the dataset for faster experimentation.
    4. Separates features from labels.
    5. Encodes categorical labels into integers.
    6. Standardizes numeric features to zero mean and unit variance.
    7. Splits the data into train/test sets with stratification.

    Args:
        path (str): Path to the CICIDS-2018 dataset CSV file.
        sample_frac (float, optional): Fraction of the dataset to use (between 0 and 1).
            Defaults to 1.0 (use entire dataset).
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: (X_train, X_test, y_train, y_test), label_encoder
            - X_train (np.ndarray): Training feature matrix.
            - X_test (np.ndarray): Test feature matrix.
            - y_train (np.ndarray): Encoded training labels.
            - y_test (np.ndarray): Encoded test labels.
            - label_encoder (LabelEncoder): Fitted label encoder for decoding predictions.

    Raises:
        FileNotFoundError: If the dataset CSV file does not exist at the given path.

    Example:
        >>> (X_train, X_test, y_train, y_test), le = load_cicids_dataset("CICIDS2018.csv", sample_frac=0.3)
        >>> X_train.shape, X_test.shape
        ((10000, 78), (2500, 78))
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found at {path}")

    df = pd.read_csv(path)

    # Remove empty columns
    df = df.dropna(axis=1, how="all")

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state)

    # Separate features/labels
    X = df.drop(columns=["Label"])
    y = df["Label"]

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=random_state, stratify=y_enc
    )

    return (X_train, X_test, y_train, y_test), le
