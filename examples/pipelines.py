import os
import pathlib
import joblib
import datetime as dt
import numpy as np
import pandas as pd

# Pipelines
from sklearn.pipeline import Pipeline

# Estimators
from sklearn.ensemble import RandomForestClassifier

# Utilities
from utilities.csv import load_dataset

# Averages
from transformers.averages import SmaFeature, EwmaFeature

# Returns
from transformers.returns import LinearReturnFeature

# Bollinger bands
from transformers.bollinger import BollingBandsFeature

# ATR
from transformers.atr import AverageTrueRangeFeature

# ##################################################################
# DEMO
# ##################################################################

if __name__ == '__main__':
    #
    # Load rough dataset
    dataset = load_dataset()

    # Create some fake labels: guessing if we'll be up or down in 5 minutes
    dataset["labels"] = np.sign(dataset["close"] / dataset["close"].shift(5) - 1.0)
    dataset["labels"].fillna(0.0, inplace=True)

    # To make sure we're binary
    dataset["labels"] = np.sign(dataset["labels"] + 0.01)

    # To check that we have only 2 cats
    res = pd.factorize(dataset["labels"])

    test_ratio = 0.9
    split_index = int(test_ratio * dataset.shape[0])
    train_set, test_set = dataset[:split_index], dataset[split_index:]

    # Split test train data
    y_train, y_test = train_set.pop("labels"), test_set.pop("labels")
    X_train, X_test = train_set, test_set

    # Instantiate some transformations
    #
    # Averages
    ewma_on_close_price_only = EwmaFeature(columns=["close", ], spans=[5, 10, 30])
    sma_on_volume_related_columns = SmaFeature(columns=["quantity", "amount", "counter"], spans=[60, 360, 720])

    # Returns
    linear_return_ohlc = LinearReturnFeature(columns=["open", "high", "low", "close"], prefix="linear")

    # Bollinger bands
    bollinger_bands = BollingBandsFeature(columns=["close", "open"], spans=[30, 60, 720], std_devs=[2.0, 2.5, 3.0])

    # Average True Range
    atr_wilder = AverageTrueRangeFeature(spans=[30, 60, 720])

    # Create pipeline for our transformations
    avg_pipeline = Pipeline(steps=[
        #
        # First some features...
        #
        ("ewma_close", ewma_on_close_price_only),
        ("sma_volume", sma_on_volume_related_columns),
        #
        ("linear_returns", linear_return_ohlc),
        #
        ("bollinger_bands", bollinger_bands),
        #
        ("atr_wilder", atr_wilder),
        #
        # ... then some model
        #
        ("classifier", RandomForestClassifier(min_samples_leaf=10)),
    ])

    # This takes about 60 seconds...
    rf_classifier = avg_pipeline.fit(X_train, y_train)
    # print(rf_classifier)

    # ...then dump it into storage
    d = dt.datetime.utcnow().strftime("%Y%m%d%H%M%S").upper()
    path_job = f"../dumps/rf_classifier_v{d}.pkl"
    os.makedirs(pathlib.Path(path_job).parent, exist_ok=True)
    joblib.dump(rf_classifier, path_job)

    # To load it back
    # rf_classifier = joblib.load(f"./dumps/rf_classifier_v{d}.pkl")

    # Make a prediction
    # train_predictions = rf_classifier.predict(X_train)
