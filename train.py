import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os

# ✅ Main dataset URL (tested and works)
DATA_URL = "https://raw.githubusercontent.com/murpi/wilddata/master/coffee.csv"
LOCAL_FILE = "coffee.csv"  # fallback path


def roast_category(roast_name: str) -> int:
    """Convert roast type into a numeric category."""
    if not isinstance(roast_name, str):
        return None
    roast_name = roast_name.strip().lower()
    mapping = {"light": 0, "medium": 1, "dark": 2}
    return mapping.get(roast_name, None)


def load_and_prepare() -> pd.DataFrame:
    """Try loading dataset from URL; fallback to local file."""
    try:
        df = pd.read_csv(DATA_URL)
        print(f"✅ Loaded dataset from URL ({len(df)} rows)")
    except Exception as e:
        print("⚠️  Could not load remote data, trying local file instead...")
        try:
            df = pd.read_csv(LOCAL_FILE)
            print(f"✅ Loaded local dataset ({len(df)} rows)")
        except Exception as e2:
            print("❌ Could not load either remote or local dataset.")
            print("   Please make sure coffee.csv is in your project folder.")
            raise e2
    return df


def train_linear_regression(df: pd.DataFrame):
    """Train Linear Regression using 100g_USD → rating."""
    X = pd.to_numeric(df["100g_USD"], errors="coerce").values.reshape(-1, 1)
    y = pd.to_numeric(df["rating"], errors="coerce")

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    pipe.fit(X, y)
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, "models/model_1.pickle")
    print("✅ model_1.pickle saved.")
    return pipe


def train_tree_regressor(df: pd.DataFrame):
    """Train Decision Tree Regressor using 100g_USD & roast_cat."""
    df["roast_cat"] = df["roast"].apply(roast_category)
    X = df[["100g_USD", "roast_cat"]].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df["rating"], errors="coerce")

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", DecisionTreeRegressor(max_depth=5, random_state=42))
    ])

    pipe.fit(X, y)
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, "models/model_2.pickle")
    print("✅ model_2.pickle saved.")
    return pipe


def predict_with_linear(bundle_path: str, df: pd.DataFrame):
    """Predict ratings using Linear Regression model."""
    pipe = joblib.load(bundle_path)
    X = pd.to_numeric(df["100g_USD"], errors="coerce").values.reshape(-1, 1)
    return pipe.predict(X)


def predict_with_tree(bundle_path: str, df: pd.DataFrame):
    """Predict ratings using Decision Tree model."""
    pipe = joblib.load(bundle_path)
    X = df[["100g_USD", "roast_cat"]].apply(pd.to_numeric, errors="coerce")
    return pipe.predict(X)


def main():
    df = load_and_prepare()
    train_linear_regression(df)
    train_tree_regressor(df)


if __name__ == "__main__":
    main()
