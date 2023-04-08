# Documentation


Author: [Yang Lin](https://yanglin1997.github.io/)

E-mail: linyang1997@yahoo.com.au

The MLOps on AWS Sagemaker is developed, with considering an enterprise machine learning system and addressing the requirements of ML Engineering Challenge. 

Highlishts of the design:
- Use of IaC for end-to-end code CICD build, pipelines build and model deployment.
- Re-package of model as container for Asynchronous (the batch transform pipeline) and Synchronous (the 'stream' and 'batch' endpoints as request) inference.
- Versioning and monitoring code, pipeline and model.
- Enable data capture and API security and monitoring for the real-time endpoint.

The Architecture diagram:

Response to the requirement:

1. the Architecture diagram:
<!-- ![img/MLOps.png](img/MLOps.png) -->
<div style="text-align:center"><img src ="img/MLOps.png" ,width=100/></div>

2. linear regression model

Vectorization is applied for both gradient descent and prediction, see 'pipelines/LR/train.py'.
time and memory complexity
Final result gives a R2 score of 0.47.
<div style="text-align:center"><img src ="img/eval_result.png" ,width=100/></div>

3. an API for inference

The Cloudformation is employed to deploy the lastes approved model. The 'staging' model is deployed first and once its approved manually, the same model would be deployed on better computational resource for 'production'. Please see 'https://github.com/YangLIN1997/model-deploy' repository. 

- `POST /stream` : it can be invoked with 'https://lbe1si5il9.execute-api.ap-southeast-2.amazonaws.com/production/stream'. It takes single record e.g. json.dumps({"x": [[0.3]]}), and returns a prediction, e.g. 434.08.
- `POST /batch` : it can be invoked with 'https://lbe1si5il9.execute-api.ap-southeast-2.amazonaws.com/production/batch'. It takes multiple records e.g. json.dumps({"x": [[0.3],[0.3]]}), and returns a prediction, e.g.  [[434.0765649977469], [434.0765649977469]]. 
For both API, the key ("x-api-key") is "YangLinYangLinYangLin". 

For the use case study, the python script 'invoke_api.py' proviudes examples of invoking and testing both endpoints. Note the packages would be auto implemented by 'subprocess' for simplicity.
Run the script 'python4 invoke_api.py', the following result is expected to see:
<div style="text-align:center"><img src ="img/API_test.png" ,width=100/></div>

Additionally, a batch inference that cable to make prediction for a batch of files peoridically is also necessary. Hence a batch transform pipeline is build with lates approved model and cable to make predictions. It is triggered with EventTrail and EventBridge whenever a new file is uploaded (real-word usually schedule the task instead), then the pipeline is going to make batch transform for all files in that folder.
Below is the EventBridge rule:
<div style="text-align:center"><img src ="img/batch_trigger_eventbridge.png" ,width=100/></div>
Below is the batch inference input file location:
<div style="text-align:center"><img src ="img/batch_trigger_in.png" ,width=100/></div>
Below is the batch inference pipeline (it is triggered when 'X_test.csv' was uploaded):
<div style="text-align:center"><img src ="img/batch_pipeline.png" ,width=100/></div>
Below is the batch inference output file location (they are transformed when 'X_test.csv' was uploaded):
<div style="text-align:center"><img src ="img/batch_trigger_out.png" ,width=100/></div>

4. package code into a python package

Yes, the 'train.py' is packaged. Also the 'dill' package is used to package trained model with object information. 

5. package code into a container

Yes, all steps in the pipeline are containerized.

6. CICD

Yes.

7. componenets for an enterprise machine learning system

Except for the highlights listed at the beginning, the system still can be improved with the following points: 

- Critical to have: 

1. monitoring system for DS to take action when there is a drift in data distribtuion and model performance metrics and bias. Once these functions are implemented, alerts can be set with SNS to notify DS/MLE, then they could considering retrain model, reviewing the system and data, without manually monitoring. Hence, one DS/MLE could have mutiple ML/DS product/pipeline under control easily. 
2. step to normalise data, feature engineering and tune hyperparameters in the ML pipeline.
3. A/B test for models with production traffic.

- Nice to have: 
1. deploy multiple models ensemble as the endpoint to improve performance.
2. model compression, e.g. quantization and distillation for deep learning.
3. half precision training for large model.
4. more feature engineering and consider statiuc features for the system.
5. optimization in the data side for large scale data system, may using database or other data file format.

8. end to end

Yes. Everytime DS/ML makes a commit, the code pipelines auto build and deploy new ML pipelines with test. And the new production endpoint would be approved manually.

9. unit tests or integration tests

Yes. For the endpoint deployment, test is employed on the 'staging' endpoint to check if it is in service, has data capture enbaled and able to be invoked. 

10. security/monitoring

Yes, code, data and models are private. Monitoring is enable with logs and CloudWatch. 
API endpoint requires the API key and the post usage is also limited. Below is the request limit on APIGateWay: 
<div style="text-align:center"><img src ="img/APIGateWay.png" ,width=100/></div>


11. document

Yes. It is documented with AWS architecture diagram and highlights and screenshots to explain architecture of MLOps on AWS, README to explain code structure, comments and logs in Codes to explain steps and functionalities.  

12. service

From the code perspective, it is well packaged and documented.
From the scalability perspective, it is developed on the Sagemaker plantform with versioning and monitoring ability and is good for multiple DS/MLE to work together. The training and deployment can also be scaled up with more instance and data parallel for training and auto-scaling policy for endpoints.

13. production ML system
This system was designed with the consideration of a production system. 
It allows good interactions between MLE/DS with ability to provide auto CICD for the code and ML pipelines on AWS, and easy for MLE/DS to scale up the system for training and model endpoint with auto-scaling policy.  
It integrates multiple AWS services smoothly and most of them are deployed with IaC. 


13. screenshot to demonstrate the system:
ML pipeline:
<div style="text-align:center"><img src ="img/ML_pipeline.png" ,width=100/></div>

ML pipeline trigger rule:
<div style="text-align:center"><img src ="img/ML_pipeline_schedule.png" ,width=100/></div>

Model-build (this CI repository) code build pipeline:
<div style="text-align:center"><img src ="img/modelbuild_pipeline.png" ,width=100/></div>

Model-deploy (another CD repository 'https://github.com/YangLIN1997/model-deploy') code build pipeline:
<div style="text-align:center"><img src ="img/modeldeploy_pipeline.png" ,width=100/></div>

Model-deploy (another CD repository 'https://github.com/YangLIN1997/model-deploy') code build is triggered by new approved model:
<div style="text-align:center"><img src ="img/modeldeploy_rule.png" ,width=100/></div>

ML model version:
<div style="text-align:center"><img src ="img/model_version.png" ,width=100/></div>

Datacapture for real-time endpoint:
<div style="text-align:center"><img src ="img/datacapture.png" ,width=100/></div>

Auto-scaling for real-time endpoint:
<div style="text-align:center"><img src ="img/autoscaling.png" ,width=100/></div>

Unit test for model-deploy:
<div style="text-align:center"><img src ="img/modeldeploy_test.png" ,width=100/></div>


## Layout of the SageMaker ModelBuild Project

This is the code repository as part of a Project in SageMaker as the CI part of the MLOps. 

```
|-- codebuild-buildspec.yml
|-- CONTRIBUTING.md
|-- pipelines
|   |-- LR
|   |   |-- __init__.py
|   |   |-- train.py
|   |   |-- evaluate.py
|   |   |-- inference.py
|   |   |-- pipeline.py
|   |   |-- pipeline_batch.py
|   |   `-- preprocess.py
|   |-- get_pipeline_definition.py
|   |-- __init__.py
|   |-- run_pipeline.py
|   |-- _utils.py
|   `-- __version__.py
|-- README.md
|-- setup.cfg
|-- setup.py
|-- tests
|   `-- test_pipelines.py
|-- invoke_api.py
`-- tox.ini
```

## Start here (readme file)

The following section provides an overview of how the code is organized and what you need to modify. In particular, `pipelines/pipelines.py` contains the core of the business logic for this problem. It has the code to express the ML steps involved in generating an ML model. You will also find the code for that supports preprocessing, training and evaluation steps in `preprocess.py` , `train.py` and `evaluate.py` files respectively. Additionally, we pack `inference.py` with pre-trained model data as a container to register it as a model on Sagemaker for inference, `pipelines/pipelines_batch.py` defines the batch inference pipeline. Note 'invoke_api.py' provides examples of invoking the 'stream' and 'batch' endpoints.

A description of some of the artifacts is provided below:
<br/><br/>
The codebuild execution instructions. 

```
|-- codebuild-buildspec.yml
```

<br/><br/>
The pipeline artifacts, which includes a pipeline module defining the required `get_pipeline` method that returns an instance of a SageMaker pipeline, a preprocessing script that is used in feature engineering, and a model evaluation script to measure the Mean Squared Error of the model that's trained by the pipeline. Note that the 'pipeline_batch.py' creates the batch inference pipeline that transforms the csv files in "s3://sagemaker-project-p-unfbxwxpd29d/batchdata/input/" and save the outputs at "s3://sagemaker-project-p-unfbxwxpd29d/batchdata/output/".

```
|-- pipelines
|   |-- LR
|   |   |-- __init__.py
|   |   |-- pipeline.py
|   |   |-- pipeline_batch.py
|   |   ```

```
<br/><br/>
Utility modules for getting pipeline definition jsons and running pipelines (you do not typically need to modify these):

```
|-- pipelines
|   |-- get_pipeline_definition.py
|   |-- __init__.py
|   |-- run_pipeline.py
|   |-- _utils.py
|   `-- __version__.py
```
<br/><br/>
Python package artifacts:
```
|-- setup.cfg
|-- setup.py
```
<br/><br/>
A stubbed testing module for testing your pipeline as you develop:
```
|-- tests
|   `-- test_pipelines.py
```
<br/><br/>
The `tox` testing framework configuration:
```
`-- tox.ini
```
