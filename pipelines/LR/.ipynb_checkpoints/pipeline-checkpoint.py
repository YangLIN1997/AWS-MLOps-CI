"""Workflow pipeline script for Aiabetes pipeline.
                                                                                 . -ModelStep
                                                                                .
    Process-> DataQualityCheck/DataBiasCheck -> Train -> Evaluate -> Condition .
                                                                                 .
                                                                                   . -(stop)
                                                  


Implements a get_pipeline(**kwargs) method.
"""
import os

import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput, CreateModelInput, TransformInput
from sagemaker.model import Model
from sagemaker.transformer import Transformer
from sagemaker.tuner import HyperparameterTuner

from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
    FileSource
)
from sagemaker.drift_check_baselines import DriftCheckBaselines
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.functions import (
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterBoolean,
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    TuningStep,
    CreateModelStep,
    TransformStep,
    CacheConfig
)
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.clarify_check_step import (
    DataBiasCheckConfig,
    ClarifyCheckStep,
    ModelBiasCheckConfig,
    ModelPredictedLabelConfig,
    ModelExplainabilityCheckConfig,
    SHAPConfig
)
from sagemaker.workflow.quality_check_step import (
    DataQualityCheckConfig,
    ModelQualityCheckConfig,
    QualityCheckStep,
)
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join
from sagemaker.model_monitor import DatasetFormat, model_monitoring
from sagemaker.clarify import (
    BiasConfig,
    DataConfig,
    ModelConfig
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
        """
     boto_session = boto3.Session(region_name=region)
     sagemaker_client = boto_session.client("sagemaker")
     return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_name=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.describe_project(ProjectName=sagemaker_project_name)
        sagemaker_project_arn = response["ProjectArn"]
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    role=None,
    default_bucket=None,
    model_package_group_name="AiabetesPackageGroup",
    pipeline_name="AiabetesPipeline",
    base_job_prefix="Aiabetes",
    processing_instance_type="ml.m5.xlarge",
    training_instance_type="ml.m5.xlarge",
    sagemaker_project_name=None,
):
    """Gets a SageMaker ML Pipeline instance working with on Aiabetes data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    default_bucket = sagemaker_session.default_bucket()
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = get_pipeline_session(region, default_bucket)
    
    # only enable cache during testing/developing the pipeline, have to disable it for production otherwise steps are skipped
    cache_config = CacheConfig(enable_caching=True, expire_after="90d")
    
    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    model_approval_status = ParameterString(
        # name="ModelApprovalStatus", default_value="PendingManualApproval"
        name="ModelApprovalStatus", default_value="Approved"
    )
    
    # not using input csv, we use sklearn to download data instead
    input_data = ParameterString(
        name="InputDataUrl",
        default_value="s3://dataset.csv",
    )

    # processing step for feature engineering
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name="data-preprocess-run",
        sagemaker_session=pipeline_session,
        role=role,
    )

    step_args = sklearn_processor.run(
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            # ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),job_name = "data-preprocess-run",
        # arguments=["--input-data", input_data],
    ) 

    step_process = ProcessingStep(
        name="PreprocessDiabetesData",
        step_args=step_args,       
        cache_config=cache_config
    )


    model_path = f"s3://{default_bucket}/{base_job_prefix}/DiabetesTrain"
    image_uri = sagemaker.image_uris.retrieve(
        framework="sklearn",
        region=region,
        version="0.23-1",
        py_version="py3",
        instance_type=training_instance_type,
    )
    # https://github.com/aws/sagemaker-python-sdk/issues/1902

    LR_train = Estimator(entry_point='train.py',
                           source_dir = 'pipelines/LR',
                          dependencies =None,
                        image_uri=image_uri,
                        instance_type=training_instance_type,
                        instance_count=1,
                        output_path=model_path,
                        base_job_name="Diabetes-train",
                        sagemaker_session=pipeline_session,
                        role=role,
                    )
    
    
#     LR_train.set_hyperparameters(
#         lr=1e-1,
#         iterations=1.5e4,
#     )

#     hyperparameter_ranges = {
#         "alpha": ContinuousParameter(0.001, 0.1, scaling_type="Logarithmic"),
#         "iterations": ContinuousParameter(0.01, 10, scaling_type="Logarithmic"),
#     }
    
#     tuner = HyperparameterTuner(estimator = LR_train, 
#                                 objective_metric_name ="validation:rmse" ,
#                                 hyperparameter_ranges=hyperparameter_ranges,
#                                 max_jobs=3,
#                                 max_parallel_jobs=3,
#                                 strategy="Random",
#                                 objective_type="Minimize",
#                                 sagemaker_session=PipelineSession()
#                                )

#     hpo_args = tuner_log.fit(
#         inputs={
#             "train": TrainingInput(
#                 s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
#                 content_type="text/csv",
#             ),
#             "validation": TrainingInput(
#                 s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
#                     "valid"
#                 ].S3Output.S3Uri,
#                 content_type="text/csv",
#             ),
#         }
#     )

#     step_tuning = TuningStep(
#         name="HPTuning",
#         step_args=hpo_args,
#         cache_config=cache_config,
#     )
    
    step_args = LR_train.fit(
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },job_name = "Diabetes-train",
    )

    step_train = TrainingStep(
        name="TrainAiabetesModel",
        step_args=step_args,
        cache_config=cache_config,
    )


    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name="script-Aiabetes-eval",
        sagemaker_session=pipeline_session,
        role=role,
    )
    step_args = script_eval.run(
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            )
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
        code=os.path.join(BASE_DIR, "evaluate.py"),
        job_name="script-Aiabetes-eval",
    )
    evaluation_report = PropertyFile(
        name="AiabetesEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    step_eval = ProcessingStep(
        name="EvaluateAiabetesModel",
        step_args=step_args,
        cache_config=cache_config,
        property_files=[evaluation_report],
    )


    ### Register the model

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json"
        )
    )
    
    model = Model(
        image_uri=image_uri,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=pipeline_session,
        source_dir = 'pipelines/LR',
        entry_point='inference.py',
        role=role,
    )

    step_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics
    )

    step_register = ModelStep(
        name="RegisterAiabetesModel",
        step_args=step_args,
    )

    # condition step for evaluating model quality and branching execution
    cond_lte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="regression_metrics.r2.value"
        ),
        right=0.2
    )
    step_cond = ConditionStep(
        name="Checkr2AiabetesEvaluation",
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[],
    )

    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            input_data,
        ],
        steps=[step_process, 
               step_train, 
               step_eval, 
               step_cond],
        sagemaker_session=pipeline_session,
    )
    return pipeline
