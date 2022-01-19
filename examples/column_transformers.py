import joblib
import datetime as dt
import numpy as np
import pandas as pd

import sklearn
SK_VERSION = sklearn.__version__

if int(SK_VERSION.split(".")[1]) > 21:
    print(f"WARNING: Cloning issue reported in Scikit-Learn base file ('is not' test failing).")
    raise ValueError(f"Scikit-Learn version is not suitable.")

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Estimators
from sklearn.ensemble import RandomForestClassifier

# Utilities
from utilities.csv import load_dataset

# Averages
from transformers.averages import SmaFeature, EwmaFeature

# Returns
from transformers.returns import LinearReturnFeature

RND_STATE = 42

if __name__ == '__main__':
    #
    # Load rough dataset
    dataset = load_dataset()

    # Create some fake labels: guessing if we'll be up or down in 5 minutes
    labels = np.sign(dataset["close"] / dataset["close"].shift(5) - 1.0)
    labels.fillna(0.0, inplace=True)

    # To make sure we're binary
    labels = np.sign(labels + 0.01)

    # To check that we have only 2 cats
    res = pd.factorize(labels)

    # Split test train data
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.10, random_state=RND_STATE)

    # Instantiate some transformations
    #
    # Averages (no columns specified)
    ewma_average = EwmaFeature(spans=[5, 10, 30])
    # sma_average = SmaFeature(spans=[60, 360, 720])

    columns_averages = ["close", "amount"]

    # Create pipeline for our transformations
    pipe_averages = Pipeline(steps=[
        ("ewma_close", ewma_average),
        # ("sma_volume", sma_average),
    ])

    # Returns
    linear_returns = LinearReturnFeature(prefix="linear")
    columns_returns = ["open", "high", "low", "close"]

    pipe_returns = Pipeline(steps=[
        ("linear_returns", linear_returns),
    ])

    pre_processor = ColumnTransformer(
        transformers=[
            ('averages', pipe_averages, columns_averages),
            ('returns', pipe_returns, columns_returns),
        ]
    )

    full_pipeline = Pipeline(
        steps=[
            ('pre_processor', pre_processor),
            ('classifier', RandomForestClassifier()),
        ]
    )

    # This takes about 60 seconds...
    rf_classifier = full_pipeline.fit(X_train, y_train)
    # print(rf_classifier)

    # then dump it into storage
    d = dt.datetime.utcnow().strftime("%Y%m%d%H%M%S").upper()
    joblib.dump(rf_classifier, f"./dumps/rf_classifier_v{d}.pkl")

    # To load it back
    # rf_classifier = joblib.load(f"./dumps/rf_classifier_v{d}.pkl")

    # Make a prediction
    # train_predictions = rf_classifier.predict(X_train)
    # score = roc_auc_score(y_train, train_predictions)
