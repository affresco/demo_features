import joblib
import datetime as dt
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Estimators
from sklearn.ensemble import RandomForestClassifier

# Utilities
from utilities.csv import load_dataset

# Averages
from averages import SmaFeature, EwmaFeature

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
    ewma_on_close_price_only = EwmaFeature(columns=["close", ], spans=[5, 10, 30])
    sma_on_volume_related_columns = SmaFeature(columns=["quantity", "amount", "counter"], spans=[60, 360, 720])

    # Create pipeline for our transformations
    avg_pipeline = Pipeline(steps=[
        ("ewma_close", ewma_on_close_price_only),
        ("sma_volume", sma_on_volume_related_columns),
        ("classifier", RandomForestClassifier(min_samples_leaf=10)),
    ])

    # This takes about 60 seconds...
    rf_classifier = avg_pipeline.fit(X_train, y_train)
    print(rf_classifier)

    # then dump it into storage
    d = dt.datetime.utcnow().strftime("%Y%m%d%H%M%S").upper()
    joblib.dump(rf_classifier, f"./dumps/rf_classifier_v{d}.pkl")

    # To load it back
    # rf_classifier = joblib.load(f"./dumps/rf_classifier_v{d}.pkl")

    # Make a prediction
    train_predictions = rf_classifier.predict(X_train)
    score = roc_auc_score(y_train, train_predictions)
