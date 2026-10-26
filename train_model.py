"""Train a reproducible demo credit-risk model compatible with main.py's request schema."""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

RANDOM_STATE = 42
OUT = Path(__file__).resolve().parent / "champion_model.joblib"

def make_demo_data(n=1200):
    rng = np.random.default_rng(RANDOM_STATE)
    age = rng.integers(21, 70, n).astype(float)
    income = np.clip(rng.normal(55000, 18000, n), 12000, 150000)
    loan_amount = np.clip(rng.normal(18000, 9000, n), 1000, 60000)
    credit_score = np.clip(rng.normal(660, 75, n), 300, 850)
    employment_years = np.clip(rng.normal(7, 5, n), 0, 40)
    education = rng.choice(["High School", "Bachelor", "Master", "Other"], n)
    home_ownership = rng.choice(["Rent", "Own", "Mortgage"], n)
    loan_purpose = rng.choice(["Car", "Education", "Medical", "Home", "Other"], n)
    risk = (
        1.2 * (loan_amount / income)
        - 0.012 * (credit_score - 650)
        + 0.025 * (age < 25)
        + 0.15 * (home_ownership == "Rent")
        + rng.normal(0, 0.35, n)
    )
    prob = 1 / (1 + np.exp(-risk))
    default = (rng.random(n) < prob).astype(int)
    df = pd.DataFrame({
        "age": age,
        "annual_income": income,
        "loan_amount": loan_amount,
        "credit_score": credit_score,
        "employment_years": employment_years,
        "education": education,
        "home_ownership": home_ownership,
        "loan_purpose": loan_purpose,
        "default": default
    })
    for col in ["annual_income", "credit_score", "employment_years", "education", "home_ownership"]:
        rows = rng.choice(df.index, size=int(0.05 * n), replace=False)
        df.loc[rows, col] = np.nan
    return df

def main():
    df = make_demo_data()
    X = df.drop(columns="default")
    y = df["default"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])
    configs = {
        "LogisticRegression": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
            {"model__C": [0.1, 1.0, 10.0]}
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
            {"model__n_estimators": [100, 200], "model__max_depth": [None, 8]}
        ),
        "ExtraTrees": (
            ExtraTreesClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
            {"model__n_estimators": [100, 200], "model__max_depth": [None, 8]}
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {"model__n_estimators": [100, 150], "model__learning_rate": [0.05, 0.1]}
        )
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    best_search = None
    best_name = None
    for name, (estimator, grid) in configs.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        score = roc_auc_score(y_test, search.best_estimator_.predict_proba(X_test)[:, 1])
        print(f"{name}: CV ROC-AUC={search.best_score_:.4f}; test ROC-AUC={score:.4f}")
        if best_search is None or search.best_score_ > best_search.best_score_:
            best_search, best_name = search, name
    joblib.dump(best_search.best_estimator_, OUT)
    print(f"Saved champion ({best_name}) to {OUT}")

if __name__ == "__main__":
    main()
