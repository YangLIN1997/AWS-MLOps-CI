import subprocess
import requests
import json
try:
    import numpy as np
except:
    subprocess.call(['pip', 'install', 'numpy', '--upgrade'])
    import numpy as np
try:
    import pandas as pd
except:
    subprocess.call(['pip', 'install', 'pandas', '--upgrade'])
    import pandas as pd
try:
    from sklearn.metrics import r2_score
    from sklearn.datasets import load_diabetes
except:
    subprocess.call(['pip', 'install', 'scikit-learn', '--upgrade'])
    from sklearn.metrics import r2_score
    from sklearn.datasets import load_diabetes

url_stream = 'https://lbe1si5il9.execute-api.ap-southeast-2.amazonaws.com/production/stream'
url_batch = 'https://lbe1si5il9.execute-api.ap-southeast-2.amazonaws.com/production/batch'
x_api_key = "YangLinYangLinYangLin"

def Average(lst):
    """
    Calculate list average value.
    :return mean of the list
    """
    return sum(lst) / len(lst)

def batch_data_prepare():
    """
    Generates a random dataset and save it.

    """
    print('Start generate dataset.')

    # Load the diabetes dataset
    diabetes_X, diabetes_y = load_diabetes(return_X_y=True)

    # Use only one feature
    diabetes_X = diabetes_X[:, np.newaxis, 2]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes_y[:-20].reshape(-1,1)
    diabetes_y_test = diabetes_y[-20:].reshape(-1,1)

    pd.DataFrame(diabetes_X_test,columns=['x']).to_csv('X_test.csv',index=False)
    pd.DataFrame(diabetes_y_test,columns=['y']).to_csv('y_test.csv',index=False)
    print('Dataset saved.')
    return

def stream_test():
    print(f"Start testing stream endpoint")
    # post the data point with api key and receive response
    query = json.dumps({"x": [[0.3]]})
    response = requests.post(url_stream, data=query, timeout=5,headers={"x-api-key":x_api_key})
    print(f"Given input {0.3}, get prediction {response.json():.2f}")

    print(f"Start testing stream endpoint response time")
    request_time = []
    # test inference speed
    for i in range(5):
        response = requests.post(url_stream,
                             data = query, timeout=5,headers={"x-api-key":x_api_key})
        response.raise_for_status()
        request_time_s = response.elapsed.total_seconds()
        request_time.extend([request_time_s])
    print("Request completed in {0:.2f}ms in average for 5 runs".format((Average(request_time)*1000)))

def batch_test():
    try:
        print(f"Start testing batch endpoint with testing data")
        # prepare batch inference data
        batch_data_prepare()
        # post multiple data points with api key and receive response and print r2 score
        df= pd.read_csv("X_test.csv")
        query = json.dumps({"x": df.values.tolist()})
        response = requests.post(url_batch, data=query, timeout=5,headers={"x-api-key":x_api_key})
        y_hat = pd.DataFrame(response.json())
        y= pd.read_csv("y_test.csv")
        r2 = r2_score(y,y_hat)
        print(f"r2_score on testing set is {r2:.2}")
    except:
        print(f"Failed testing batch endpoint with testing data")
    print(f"Start testing batch endpoint with user-defined data")
    # post multiple data points with api key and receive response
    query = json.dumps({"x": [[0.3],[0.3]]})
    response = requests.post(url_batch, data=query, timeout=5,headers={"x-api-key":x_api_key})
    print(f"Given input {[[0.3],[0.3]]}, get prediction {response.json()}")

if __name__ == "__main__":
    stream_test()
    print('\n')
    batch_test()
