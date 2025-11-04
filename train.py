"""
train.py

Trains two models on the coffee dataset and saves them as:
 - model_1.pickle  (LinearRegression on 100g_USD -> rating)
 - model_2.pickle  (DecisionTreeRegressor on [100g_USD, roast_cat] -> rating)

Run:
    python train.py
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Dataset URL (replace with the provided URL if different)
DATA_URL = (
    "https://raw.githubusercontent.com/dataprofessor/data/master/coffee_ratings.csv"
)  # example; change to actual course URL if needed

OUT_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)


def roast_category(roast_value):
    """
    Map roast strings to integers.
    Missing / unknown -> np.nan to allow explicit handling later.
    Extend this mapping if you see more roast strings in the dataset.
    """
    if pd.isna(roast_value):
        return np.nan
    r = str(roast_value).strip().lower()
    mapping = {
        "light": 0,
        "light-medium": 1,
        "medium-light": 1,
        "medium": 2,
        "medium-dark": 3,
        "dark": 4,
        "espresso": 5,
    }
    # try to find an exact key or a substring match
    if r in mapping:
        return mapping[r]
    # fallback: look for key contained in string
    for key in mapping:
        if key in r:
            return mapping[key]
    # unknown roast -> np.nan (we'll impute or fill before training)
    return np.nan


def load_and_prepare(url=DATA_URL):
    df = pd.read_csv(url)
    # ensure required columns exist
    for col in ("100g_USD", "rating", "roast"):
        if col not in df.columns:
            raise KeyError(f"Required column missing from dataset: {col}")
    # create roast_cat
    df["roast_cat"] = df["roast"].apply(roast_category)
    return df


def train_linear_regression(df, out_path=OUT_DIR / "model_1.pickle"):
    # Use only rows with numeric 100g_USD and rating
    df_lr = df[["100g_USD", "rating"]].copy()
    df_lr = df_lr.replace([np.inf, -np.inf], np.nan).dropna(subset=["100g_USD", "rating"])
    X = df_lr[["100g_USD"]].values.astype(float)
    y = df_lr["rating"].values.astype(float)

    # Build simple pipeline to impute missing price if any when predicting later
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("lr", LinearRegression()),
        ]
    )
    pipeline.fit(X, y)
    joblib.dump(pipeline, out_path)
    print(f"Saved LinearRegression pipeline to {out_path}")


def train_decision_tree(df, out_path=OUT_DIR / "model_2.pickle"):
    df_dt = df[["100g_USD", "roast_cat", "rating"]].copy()
    # replace inf and drop rows missing rating
    df_dt = df_dt.replace([np.inf, -np.inf], np.nan).dropna(subset=["rating"])
    # For the decision tree, per the assignment suggestion we will **fill** missing roast_cat with -99
    # (so decision tree sees a distinct numeric value for "missing")
    X = df_dt[["100g_USD", "roast_cat"]].copy()
    # If 100g_USD is missing, impute median; roast_cat missing -> -99
    price_imputer = SimpleImputer(strategy="median")
    X_price = price_imputer.fit_transform(X[["100g_USD"]])
    roast_filled = X["roast_cat"].fillna(-99).values.reshape(-1, 1)
    X_prepared = np.hstack([X_price, roast_filled]).astype(float)

    y = df_dt["rating"].values.astype(float)

    dtr = DecisionTreeRegressor(random_state=42)
    dtr.fit(X_prepared, y)

    # Save a small object with model, and the price median so we can prepare inputs later
    saved = {
        "model": dtr,
        "price_imputer_median": float(price_imputer.statistics_[0]),
        "roast_missing_value": -99.0,
        "feature_order": ["100g_USD", "roast_cat"],
    }
    joblib.dump(saved, out_path)
    print(f"Saved DecisionTreeRegressor bundle to {out_path}")


def predict_with_linear(model_path, df_X):
    """
    df_X: DataFrame with column "100g_USD"
    Returns: predictions array
    """
    pipeline = joblib.load(model_path)
    if "100g_USD" not in df_X.columns:
        raise KeyError("Input DataFrame must have column '100g_USD'")
    X = df_X[["100g_USD"]].values.astype(float)
    return pipeline.predict(X)


def predict_with_tree(bundle_path, df_X):
    """
    df_X: DataFrame with columns ["100g_USD", "roast_cat"]
    Uses stored median and missing roast sentinel to build input for prediction.
    """
    saved = joblib.load(bundle_path)
    model = saved["model"]
    median = saved["price_imputer_median"]
    roast_missing = saved["roast_missing_value"]
    # prepare X
    if "100g_USD" not in df_X.columns or "roast_cat" not in df_X.columns:
        raise KeyError("Input DataFrame must have columns '100g_USD' and 'roast_cat'")
    price = pd.to_numeric(df_X["100g_USD"], errors="coerce").to_numpy().reshape(-1, 1)
    # impute price with median
    price = np.where(np.isfinite(price), price, median).astype(float)
    roast = df_X["roast_cat"].fillna(roast_missing).to_numpy().reshape(-1, 1)
    X = np.hstack([price, roast])
    return model.predict(X)


def main():
    df = load_and_prepare()
    train_linear_regression(df)
    train_decision_tree(df)
    # quick example predictions (so you can test locally)
    example_lr = pd.DataFrame({"100g_USD": [10.0, 15.0, 8.5]})
    lr_preds = predict_with_linear(OUT_DIR / "model_1.pickle", example_lr)
    print("Linear regression preds:", lr_preds)

    # For decision tree example: map roast strings to roast_cat using roast_category
    example_dt = pd.DataFrame(
        {
            "100g_USD": [10.0, 15.0, 8.5],
            "roast_cat": [roast_category("Medium-Light"), roast_category("Dark"), np.nan],
        }
    )
    dt_preds = predict_with_tree(OUT_DIR / "model_2.pickle", example_dt)
    print("Decision tree preds:", dt_preds)


if __name__ == "__main__":
    main()
