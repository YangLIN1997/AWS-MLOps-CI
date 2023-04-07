"""Evaluation script for measuring mean squared error."""
import json
import logging
import pathlib
import pickle
import subprocess
subprocess.call(['pip', 'install', 'dill'])
import dill
import tarfile
import numpy as np
import pandas as pd
import os
import sys
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.debug("Starting evaluation.")
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    logger.debug("Loading SimpleLinearRegression model.")
    with open('model.pkl', 'rb') as s:
        model = dill.load(s)

    logger.debug("Reading test data.")
    X_test_path = "/opt/ml/processing/test/diabetes_X_test.csv"
    y_test_path = "/opt/ml/processing/test/diabetes_y_test.csv"
    X_test = pd.read_csv(X_test_path).values.reshape(-1,1)
    y_test = pd.read_csv(y_test_path).values.reshape(-1,1)
    # df = pd.read_csv(os.path.join("/opt/ml/processing/test/test.csv"))
    # X_test = df.values[:,0].reshape(-1,1)
    # y_test = df.values[:,1].reshape(-1,1)

    logger.info("Performing predictions against test data.")
    y_hat = model.predict(X_test)

    mse = mean_squared_error(y_test, y_hat)
    logger.info(f"Mean squared error: {mse:.2f}")
    # The coefficient of determination: 1 is perfect prediction
    r2 = r2_score(y_test, y_hat)
    logger.info(f"Coefficient of determination: {r2:.2f}")
    report_dict = {
        
        "regression_metrics": {
            "mse": {
                "value": mse
            },
            "r2": {
                "value": r2
            },
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing out evaluation report with mse:  {mse:.2f}, r2: {r2:.2f}")
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
