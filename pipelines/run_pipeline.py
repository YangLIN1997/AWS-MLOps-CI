# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""A CLI to create or update and run pipelines."""
from __future__ import absolute_import

import argparse
import json
import sys
import traceback

from pipelines._utils import get_pipeline_driver, convert_struct, get_pipeline_custom_tags
# ===============

import boto3
import sagemaker
from sagemaker import get_execution_role, session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.functions import (
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,TransformStep,CreateModelStep
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.pytorch.estimator import PyTorch
from sagemaker.inputs import TransformInput, CreateModelInput
from sagemaker.workflow.steps import CacheConfig
import os
def batch_inference_pipeline(role=None):
    
    
    sm_client = boto3.client("sagemaker")
    model_package_arn=sm_client.list_model_packages(
                ModelPackageGroupName="mlops-yl-p-yu2ungcsmceg",
                ModelApprovalStatus="Approved",
                SortBy="CreationTime",
                MaxResults=100,
            )["ModelPackageSummaryList"][0]["ModelPackageArn"]
    model = sagemaker.model.ModelPackage(model_package_arn=model_package_arn,role=role)
    
    batch_input = "s3://sagemaker-project-p-yu2ungcsmceg/batch_data/X_test.csv"
    batch_output = "s3://sagemaker-project-p-yu2ungcsmceg/batch_data/"
    transform_job = model.transformer(instance_count = 1,     
                                      instance_type = 'ml.m5.large',      
                                      strategy='MultiRecord',  
                                      assemble_with='Line', 
                                      output_path = batch_output, 
                                      accept="text/csv",
                                      env = {'SAGEMAKER_MODEL_SERVER_TIMEOUT' : '3600' },
                                      max_concurrent_transforms=1, max_payload=6)    
    step_transform = TransformStep(name="batch_inference",transformer = transform_job,
                                   inputs =TransformInput(data=batch_input,data_type="S3Prefix",
                                    input_filter="$[0]",# no index
                                    # input_filter="$[1,5]",
                                    content_type="text/csv",split_type="Line")
    )
    
    create_model_pipeline = Pipeline(name="LRBatchInferencePipeline", steps=[step_transform],
            parameters=[
               batch_input,batch_output
            ],)
    
    try:
        create_model_pipeline.create(role_arn=role)
        print(f"\n###### Created DeBERTaV3BatchInferencePipeline.")
    except:
        print(f"\n###### DeBERTaV3BatchInferencePipeline was created.")
    create_model_pipeline.update(role_arn=role)

    execution = create_model_pipeline.start(execution_display_name="createBatchInferencepipeline")
    print(f"\n###### Execution started with PipelineExecutionArn: {execution.arn}")

    print("Waiting for the execution to finish...")

    # Setting the attempts and delay (in seconds) will modify the overall time the pipeline waits. 
    # If the execution is taking a longer time, update these parameters to a larger value.
    # Eg: The total wait time is calculated as 60 * 120 = 7200 seconds (2 hours)
    execution.wait(max_attempts=120, delay=60)

    print("\n#####Execution completed. Execution step details:")

    print(execution.list_steps())

    

def main():  # pragma: no cover
    """The main harness that creates or updates and runs the pipeline.

    Creates or updates the pipeline and runs it.
    """
    parser = argparse.ArgumentParser(
        "Creates or updates and runs the pipeline for the pipeline script."
    )

    parser.add_argument(
        "-n",
        "--module-name",
        dest="module_name",
        type=str,
        help="The module name of the pipeline to import.",
    )
    parser.add_argument(
        "-kwargs",
        "--kwargs",
        dest="kwargs",
        default=None,
        help="Dict string of keyword arguments for the pipeline generation (if supported)",
    )
    parser.add_argument(
        "-role-arn",
        "--role-arn",
        dest="role_arn",
        type=str,
        help="The role arn for the pipeline service execution role.",
    )
    parser.add_argument(
        "-description",
        "--description",
        dest="description",
        type=str,
        default=None,
        help="The description of the pipeline.",
    )
    parser.add_argument(
        "-tags",
        "--tags",
        dest="tags",
        default=None,
        help="""List of dict strings of '[{"Key": "string", "Value": "string"}, ..]'""",
    )
    args = parser.parse_args()

    if args.module_name is None or args.role_arn is None:
        parser.print_help()
        sys.exit(2)
    tags = convert_struct(args.tags)

    try:
        pipeline = get_pipeline_driver(args.module_name, args.kwargs)
        print("###### Creating/updating a SageMaker Pipeline with the following definition:")
        parsed = json.loads(pipeline.definition())
        print(json.dumps(parsed, indent=2, sort_keys=True))

        all_tags = get_pipeline_custom_tags(args.module_name, args.kwargs, tags)

        upsert_response = pipeline.upsert(
            role_arn=args.role_arn, description=args.description, tags=all_tags
        )
        print("\n###### Created/Updated SageMaker Pipeline: Response received:")
        print(upsert_response)

        # Parameters should be passed for the first execution of this pipeline so all
        # the baselines calculated will be registered as DriftCheckBaselines.
        # After the first execution, the default parameters can be used.

        execution = pipeline.start()

        # Update above code as below to use default parameter values for future pipeline executions
        # after approving the model registered by the first pipeline execution in Model Registry
        # so that all the checks are enabled and previous baselines are retained.

        # execution = pipeline.start()

        print(f"\n###### Execution started with PipelineExecutionArn: {execution.arn}")

        print("Waiting for the execution to finish...")

        # Setting the attempts and delay (in seconds) will modify the overall time the pipeline waits. 
        # If the execution is taking a longer time, update these parameters to a larger value.
        # Eg: The total wait time is calculated as 60 * 120 = 7200 seconds (2 hours)
        execution.wait(max_attempts=120, delay=60)
        
        print("\n#####Execution completed. Execution step details:")

        print(execution.list_steps())
        
        
        # print("###### Creating/updating a SageMaker Batch Inference Pipeline:")
        # batch_inference_pipeline(args.role_arn)
        
    except Exception as e:  # pylint: disable=W0703
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
