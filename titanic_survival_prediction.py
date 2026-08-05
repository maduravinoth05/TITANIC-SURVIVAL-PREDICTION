"""
Titanic Survival Prediction using Artificial Intelligence (Machine Learning)
------------------------------------------------------------------------------
CodTech Internship Project

Author : <Your Name>
Task   : Build a Machine Learning model that predicts whether a passenger
         survived the Titanic disaster, based on features like age, sex,
         passenger class, fare, number of relatives aboard, etc.

Models used : Logistic Regression, Random Forest Classifier
Libraries   : pandas, numpy, scikit-learn, matplotlib, seaborn
------------------------------------------------------------------------------
HOW TO RUN
    1. Place the Kaggle Titanic dataset file as  "train.csv"  in the same
       folder as this script (download it from:
       https://www.kaggle.com/competitions/titanic/data).
    2. If "train.csv" is NOT found, the script automatically generates a
       synthetic Titanic-like dataset so the project still runs end-to-end
       for demo / testing purposes.
    3. Run:   python titanic_survival_prediction.py
------------------------------------------------------------------------------
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

RANDOM_STATE = 42


# ------------------------------------------------------------------ #
# 1. LOAD DATA                                                       #
# ------------------------------------------------------------------ #
def load_data(path="train.csv"):
    """Load the Kaggle Titanic dataset; fall back to a synthetic dataset
    (same schema) if the real file isn't available, so the script always
    runs end-to-end."""
    if os.path.exists(path):
        print(f"[INFO] Loading dataset from '{path}' ...")
        return pd.read_csv(path)

    print(f"[WARNING] '{path}' not found. Generating a synthetic "
          f"Titanic-like dataset instead (for demo purposes only).")
    return generate_synthetic_titanic_data(n=891)


def generate_synthetic_titanic_data(n=891, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)

    pclass = rng.choice([1, 2, 3], size=n, p=[0.24, 0.21, 0.55])
    sex = rng.choice(["male", "female"], size=n, p=[0.65, 0.35])
    age = np.clip(rng.normal(29, 14, size=n), 0.42, 80).round(1)
    sibsp = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.68, 0.23, 0.05, 0.02, 0.02])
    parch = rng.choice([0, 1, 2, 3], size=n, p=[0.76, 0.13, 0.08, 0.03])
    fare = np.round(np.clip(rng.exponential(scale=32, size=n) + (3 - pclass) * 10, 4, 512), 2)
    embarked = rng.choice(["S", "C", "Q"], size=n, p=[0.72, 0.19, 0.09])

    # Introduce some realistic missing values
    age[rng.choice(n, size=int(n * 0.20), replace=False)] = np.nan
    embarked = embarked.astype(object)
    embarked[rng.choice(n, size=2, replace=False)] = np.nan

    # Survival probability loosely mirrors real Titanic patterns:
    # women, children, and higher class passengers survived more often.
    base = 0.5
    logit = (
        base
        + (sex == "female") * 1.6
        + (pclass == 1) * 0.9
        + (pclass == 2) * 0.3
        - (pclass == 3) * 0.5
        + (np.nan_to_num(age, nan=29) < 12) * 0.7
        - (np.nan_to_num(age, nan=29) > 60) * 0.4
        + (fare > 50) * 0.3
        - 1.0
    )
    prob = 1 / (1 + np.exp(-logit))
    survived = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame({
        "PassengerId": np.arange(1, n + 1),
        "Survived": survived,
        "Pclass": pclass,
        "Name": [f"Passenger, Mr/Mrs {i}" for i in range(n)],
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Ticket": [f"T{1000 + i}" for i in range(n)],
        "Fare": fare,
        "Cabin": np.nan,
        "Embarked": embarked,
    })
    return df


# ------------------------------------------------------------------ #
# 2. PREPROCESS DATA                                                 #
# ------------------------------------------------------------------ #
def preprocess(df):
    df = df.copy()

    # Feature engineering
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Handle missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # Encode categorical variables
    df["Sex"] = LabelEncoder().fit_transform(df["Sex"])           # male=1, female=0
    df["Embarked"] = LabelEncoder().fit_transform(df["Embarked"])  # S/C/Q -> 0/1/2

    features = ["Pclass", "Sex", "Age", "SibSp", "Parch",
                "Fare", "Embarked", "FamilySize", "IsAlone"]
    X = df[features]
    y = df["Survived"]
    return X, y, features


# ------------------------------------------------------------------ #
# 3. TRAIN & EVALUATE MODELS                                         #
# ------------------------------------------------------------------ #
def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = (model, acc, preds)

        print("\n" + "=" * 60)
        print(f"MODEL: {name}")
        print("=" * 60)
        print(f"Accuracy : {acc * 100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, preds, target_names=["Died", "Survived"]))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, preds))

    return results


def show_feature_importance(model, features):
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=features)
        importances = importances.sort_values(ascending=False)
        print("\nFeature Importance (Random Forest):")
        for feat, score in importances.items():
            bar = "#" * int(score * 100)
            print(f"  {feat:<12}: {score:.4f}  {bar}")


def predict_sample_passenger(model, features):
    """Demonstrate a single prediction for a hand-crafted passenger."""
    sample = pd.DataFrame([{
        "Pclass": 1,
        "Sex": 0,      # female
        "Age": 28,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 80.0,
        "Embarked": 0,  # S
        "FamilySize": 1,
        "IsAlone": 1,
    }])[features]

    pred = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0][1]
    outcome = "SURVIVED" if pred == 1 else "DID NOT SURVIVE"
    print("\n" + "-" * 60)
    print("SAMPLE PREDICTION -> 1st Class, Female, Age 28, Fare 80")
    print(f"Predicted outcome : {outcome}")
    print(f"Survival probability: {prob * 100:.2f}%")
    print("-" * 60)


# ------------------------------------------------------------------ #
# MAIN                                                                #
# ------------------------------------------------------------------ #
def main():
    print("#" * 60)
    print("   TITANIC SURVIVAL PREDICTION - AI/ML PROJECT")
    print("#" * 60)

    df = load_data("train.csv")
    print(f"\n[INFO] Dataset shape: {df.shape}")
    print("\n[INFO] First 5 rows:")
    print(df.head())

    print("\n[INFO] Missing values per column:")
    print(df.isnull().sum())

    X, y, features = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n[INFO] Training samples: {X_train.shape[0]}  |  Testing samples: {X_test.shape[0]}")

    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    best_name = max(results, key=lambda k: results[k][1])
    best_model = results[best_name][0]
    print(f"\n[RESULT] Best performing model: {best_name} "
          f"({results[best_name][1] * 100:.2f}% accuracy)")

    show_feature_importance(results["Random Forest"][0], features)
    predict_sample_passenger(best_model, features)

    print("\n[INFO] Done. Project executed successfully.")


if __name__ == "__main__":
    main()
