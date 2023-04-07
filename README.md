## Layout of the SageMaker ModelBuild Project Template

The template provides a starting point for bringing your SageMaker Pipeline development to production.

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
`-- tox.ini
```

## Start here
This is the code repository as part of a Project in SageMaker as the CI part of the MLOps. 

The following section provides an overview of how the code is organized and what you need to modify. In particular, `pipelines/pipelines.py` contains the core of the business logic for this problem. It has the code to express the ML steps involved in generating an ML model. You will also find the code for that supports preprocessing, training and evaluation steps in `preprocess.py` , `train.py` and `evaluate.py` files respectively. Additionally, we pack `inference.py` with pre-trained model data as a container to register it as a model on Sagemaker for inference, `pipelines/pipelines_batch.py` defines the batch inference pipeline.

A description of some of the artifacts is provided below:
<br/><br/>
The codebuild execution instructions. 

```
|-- codebuild-buildspec.yml
```

<br/><br/>
The pipeline artifacts, which includes a pipeline module defining the required `get_pipeline` method that returns an instance of a SageMaker pipeline, a preprocessing script that is used in feature engineering, and a model evaluation script to measure the Mean Squared Error of the model that's trained by the pipeline. 

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
