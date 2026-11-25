
import time
import shutil
from pathlib import Path

import pandas as pd
import joblib
from pycaret.classification import (
    setup,
    compare_models,
    pull,
    plot_model
)

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)
from lightgbm import LGBMClassifier


def main():

    Path("images").mkdir(exist_ok=True)

    df = pd.read_csv("data/bank.csv")

    # ==========================================
    # PART 1: PYCARET WORKFLOW
    # ==========================================

    print("\n========== PYCARET WORKFLOW ==========\n")

    pycaret_start = time.perf_counter()

    setup(
        data=df,
        target="y",
        session_id=42,
        train_size=0.7,
        fold=10,
        normalize=True,
        normalize_method="zscore",
        n_jobs=1,
        html=False,
        verbose=False
    )

    top_models = compare_models(
        n_select=3,
        sort="Accuracy"
    )

    comparison_table = pull()

    pycaret_time = time.perf_counter() - pycaret_start

    print("\nPYCARET COMPARISON TABLE:\n")
    print(comparison_table.to_string())

    comparison_table.to_csv(
        "images/pycaret_results.csv"
    )

    print("\nBest PyCaret model:")
    print(type(top_models[0]).__name__)

    matrix_path = plot_model(
        top_models[0],
        plot="confusion_matrix",
        save=True
    )

    shutil.move(
        matrix_path,
        "images/confusion_matrix.png"
    )

    # ==========================================
    # PART 2: SCIKIT-LEARN WORKFLOW
    # ==========================================

    print("\n========== SCIKIT-LEARN WORKFLOW ==========\n")

    sklearn_start = time.perf_counter()

    supported_models = (
        LogisticRegression,
        RidgeClassifier,
        KNeighborsClassifier,
        DecisionTreeClassifier,
        RandomForestClassifier,
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        AdaBoostClassifier,
        GaussianNB,
        SVC,
        LinearSVC,
        LinearDiscriminantAnalysis,
        QuadraticDiscriminantAnalysis,
        LGBMClassifier
    )

    selected_model = next(
        (
            model for model in top_models
            if isinstance(model, supported_models)
        ),
        None
    )

    if selected_model is None:
        raise ValueError(
            "No supported model found in the top three."
        )

    estimator = clone(selected_model)

    if "random_state" in estimator.get_params():
        estimator.set_params(random_state=42)

    print("Selected model:", type(estimator).__name__)

    X = df.drop(columns=["y"])
    y = df["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=42
    )

    numeric_features = X.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude="number"
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", estimator)
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    print("\nCLASSIFICATION REPORT:\n")
    print(report)

    joblib.dump(model, "model.joblib")

    sklearn_time = time.perf_counter() - sklearn_start

    # ==========================================
    # PART 3: SAVE RESULTS
    # ==========================================

    print("\n========== FINAL RESULTS ==========\n")

    print(f"PyCaret time: {pycaret_time:.2f} seconds")
    print(f"Scikit-learn time: {sklearn_time:.2f} seconds")
    print("Saved model: model.joblib")

    with open("results.txt", "w") as file:
        file.write("PYCARET COMPARISON TABLE\n")
        file.write(comparison_table.to_string())
        file.write("\n\n")

        file.write(
            f"Best PyCaret model: "
            f"{type(top_models[0]).__name__}\n"
        )

        file.write(
            f"Selected scikit-learn model: "
            f"{type(estimator).__name__}\n\n"
        )

        file.write("SCIKIT-LEARN CLASSIFICATION REPORT\n")
        file.write(report)

        file.write(
            f"\nPyCaret time: {pycaret_time:.2f} seconds\n"
        )

        file.write(
            f"Scikit-learn time: {sklearn_time:.2f} seconds\n"
        )


if __name__ == "__main__":
    main()
