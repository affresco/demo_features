import logging
import datetime as dt
import pandas as pd

TIMESTAMP_COLUMN = "unix"


def load_dataset():
    #
    # Path to demo file
    path_spot = f"../data/dataset.csv"
    df = pd.read_csv(path_spot)
    logging.info(f"Spot dataset loaded.")

    # Setting the index
    idx = pd.DatetimeIndex([dt.datetime.utcfromtimestamp(d) for d in df[TIMESTAMP_COLUMN].to_numpy()])
    df.set_index(idx, inplace=True)
    logging.info(f"Dataset index set from column {TIMESTAMP_COLUMN}")

    # Sorting the index in place
    df.sort_index(inplace=True)
    logging.info(f"Dataset index sorted.")

    df.index.drop_duplicates(keep="last")
    logging.info(f"Dataset (potential) duplicates dropped.")

    df = df.resample("1min").last()
    logging.info(f"Dataset re-sampled at 1 minute interval.")

    return df
