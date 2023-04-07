"""Loader and Feature engineers the abalone dataset."""
import argparse
import logging
import os
import pathlib
import requests
import tempfile

import boto3
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def merge_two_dicts(x, y):
    """Merges two dicts, returning a new copy."""
    z = x.copy()
    z.update(y)
    return z


if __name__ == "__main__":
    logger.debug("Starting preprocessing.")

    base_dir = "/opt/ml/processing"
    
    logger.info("Load the diabetes dataset")
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

    logger.info("Use only one feature")
    diabetes_X = diabetes_X[:, np.newaxis, 2]

    logger.info("Split the data into training/testing sets")
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]
    diabetes_y_train = diabetes_y[:-20].reshape(-1,1)
    diabetes_y_test = diabetes_y[-20:].reshape(-1,1)

    logger.info("Writing out datasets to %s.", base_dir)
    pd.DataFrame(diabetes_X_train).to_csv(f"{base_dir}/train/diabetes_X_train.csv", header=False, index=False)
    pd.DataFrame(diabetes_X_test).to_csv(f"{base_dir}/test/diabetes_X_test.csv", header=False, index=False)
    pd.DataFrame(diabetes_y_train).to_csv(f"{base_dir}/train/diabetes_y_train.csv", header=False, index=False)
    pd.DataFrame(diabetes_y_test).to_csv(f"{base_dir}/test/diabetes_y_test.csv", header=False, index=False)
    
    # pd.DataFrame(np.concatenate((diabetes_X_train, diabetes_y_train), axis=1)).to_csv(f"{base_dir}/train/train.csv", header=True, index=False)
    # pd.DataFrame(np.concatenate((diabetes_X_test, diabetes_y_test), axis=1)).to_csv(f"{base_dir}/test/test.csv", header=True, index=False) 
    