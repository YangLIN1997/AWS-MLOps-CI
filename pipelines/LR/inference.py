import subprocess
subprocess.call(['pip', 'install', 'dill'])
import numpy as np
import pandas as pd
import logging
import pathlib
import os
import dill
import json
from io import StringIO

JSON_CONTENT_TYPE = 'application/json'
CSV_CONTENT_TYPE = 'text/csv'
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
 
def model_fn(model_dir):
    print("LOADING MODEL")
    print("model_dir",os.listdir(model_dir))
    with open(os.path.join(model_dir, "model.pkl"), 'rb') as s:
        model = dill.load(s)
    return model

# input_fn: Converts the incoming request payload into a numpy array.
def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    if content_type == JSON_CONTENT_TYPE:
        # input_data = json.loads(serialized_input_data)
        # df = pd.io.json.json_normalize(input_data)
        # return df['x'].values.reshape(-1,1)
        input_data = json.loads(serialized_input_data)["x"]
        df = pd.DataFrame(input_data)
        return df.values.reshape(-1,1)
    elif content_type == CSV_CONTENT_TYPE:
        df = pd.read_csv(StringIO(serialized_input_data))
        return df['x'].values.reshape(-1,1)
    else:
        raise Exception("Requested unsupported ContentType in Accept: " + content_type)
        return

# predict_fn: Performs the prediction.
def predict_fn(input_data, model):
    y_hat = model.predict(input_data)
    return y_hat.tolist()

# output_fn: Converts the prediction output into the response payload.
def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept

    elif accept == CSV_CONTENT_TYPE:
        return json.dumps({'y_hat':prediction_output})
    raise Exception("Requested unsupported ContentType in Accept: " + accept)
    
    