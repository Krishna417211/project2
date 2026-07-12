

"""
Model training module.

This module loads the cleaned taxi dataset, trains a Linear Regression
model, evaluates its performance, and saves the trained model.
"""
import os
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

FEATURES = [
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "tip_amount",
]

TARGET = "total_amount"


def load_data(path="data/taxi_clean.csv"):
    """
    Load the cleaned dataset.

    Args:
        path (str): Path to the cleaned CSV file.

    Returns:
        tuple: Feature matrix (X) and target vector (y).
    """
    df = pd.read_csv(path)
    x = df[FEATURES]
    y = df[TARGET]
    return x, y


def split_data():
    """
    Split the dataset into training and testing sets.

    Returns:
        tuple: x_train, x_test, y_train, y_test.
    """
    x, y = load_data()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    return x_train, x_test, y_train, y_test


def train_model(x_train, y_train):
    """
    Train a Linear Regression model.

    Args:
        x_train (DataFrame): Training features.
        y_train (Series): Training target.

    Returns:
        LinearRegression: Trained model.
    """
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    """
    Evaluate the trained model using the R² score.

    Args:
        model: Trained machine learning model.
        x_test (DataFrame): Test features.
        y_test (Series): Test target.

    Returns:
        float: R² score.
    """
    predictions = model.predict(x_test)
    score = r2_score(y_test, predictions)
    return score


def save_model(model, path="models/taxi_model.pkl"):
    """
    Save the trained model to disk.

    Args:
        model: Trained machine learning model.
        path (str): Output file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    joblib.dump(model, path)


def main():
    """
    Train, evaluate, and save the machine learning model.
    """
    x_train, x_test, y_train, y_test = split_data()

    model = train_model(x_train, y_train)
    score = evaluate_model(model, x_test, y_test)
    save_model(model)

    print(f"R² Score: {score:.4f}")
    print("Model saved successfully.")


if __name__ == "__main__":
    main()
