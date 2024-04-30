"""Housekeeping utilities."""

import yaml
import os


def load_config():
    _dir = os.path.abspath(os.path.dirname(__file__))
    with open(f"{_dir}/config.yml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # PROJECT = config["PROJECT"]
    # REGION = config["REGION"]
    # BUCKET = config["BUCKET"]
    # SERVICE_ACCOUNT = config["SERVICE_ACCOUNT"]
    config["PARENT"] = f'projects/{config["PROJECT"]}/locations/{config["REGION"]}'
    # VERSION = config["VERSION"]
    # DATASET_DISPLAY_NAME = config["DATASET_DISPLAY_NAME"]
    config["MODEL_DISPLAY_NAME"] = f'{config["DATASET_DISPLAY_NAME"]}-classifier-{config["VERSION"]}'
    config["WORKSPACE"] = f'gs://{config["BUCKET"]}/{config["DATASET_DISPLAY_NAME"]}'
    config["RAW_SCHEMA_DIR"] = config.get("RAW_SCHEMA_DIR", f'{config["DATASET_DISPLAY_NAME"]}/raw_schema')
    # MLMD_SQLLITE = config["MLMD_SQLLITE"]
    config["ARTIFACT_STORE"] = os.path.join(config["WORKSPACE"], 'tfx_artifacts_interactive')
    config["MODEL_REGISTRY"] = os.path.join(config["WORKSPACE"], 'model_registry')
    config["PIPELINE_NAME"] = f'{config["MODEL_DISPLAY_NAME"]}-train-pipeline'
    config["PIPELINE_ROOT"] = os.path.join(config["ARTIFACT_STORE"], config["PIPELINE_NAME"])
    config["MODULE_ROOT"] = f'gs://{config["BUCKET"]}/{config["DATASET_DISPLAY_NAME"]}/pipeline_module/{config["PIPELINE_NAME"]}'
    config["PIPELINE_DEFINITION_FILE"] = f'{config["DATASET_DISPLAY_NAME"]}.json'
    config["DATA_ROOT"] = f'gs://{config["BUCKET"]}/{config["DATASET_DISPLAY_NAME"]}/data'
    config["SERVING_MODEL_DIR"] = f'gs://{config["BUCKET"]}/{config["DATASET_DISPLAY_NAME"]}/serving_model/{config["PIPELINE_NAME"]}'
    config["MODULE_FILE_NAME"] = config["MODULE_FILE"].split('/')[-1]
    config["MODULE_PATH"] = f'gs://{config["BUCKET"]}/{config["DATASET_DISPLAY_NAME"]}/pipeline_root/{config["PIPELINE_NAME"]}/{config["MODULE_FILE_NAME"]}'
    config["MODULE_PATH_IN_BUCKET"] = f'pipeline_root/{config["PIPELINE_NAME"]}/{config["MODULE_FILE_NAME"]}'
    config["BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS"] = [
        '--project=' + config["PROJECT"],
        '--temp_location=' + os.path.join('gs://', config["BUCKET"], 'tmp'),
    ]
    # REMOVE_ARTIFACTS = config["REMOVE_ARTIFACTS"]
    # print("Project ID:", config["PROJECT"])
    # print("Region:", config["REGION"])
    # print("Bucket name:", config["BUCKET"])
    # print("Service Account:", config["SERVICE_ACCOUNT"])
    # print("Vertex API Parent URI:", config["PARENT"])
    # print("Version:", config["VERSION"])
    # print("Dataset display name:", config["DATASET_DISPLAY_NAME"])
    # print("Model display name:", config["MODEL_DISPLAY_NAME"])
    # print("Workspace:", config["WORKSPACE"])
    # print("Raw schema dir:", config["RAW_SCHEMA_DIR"])
    # print("Module file local path:", config["MODULE_FILE"])
    # print("Module file GCP path:", f'{config["MODULE_ROOT"]}/{config["MODULE_PATH_IN_BUCKET"]}')
    # print("MLMD sqllite:", config["MLMD_SQLLITE"])
    # print("Artifact store:", config["ARTIFACT_STORE"])
    # print("Model registry:", config["MODEL_REGISTRY"])
    # print("Pipeline name:", config["PIPELINE_NAME"])
    # print("Pipeline root:", config["PIPELINE_ROOT"])
    # print("Pipeline definition file:", config["PIPELINE_DEFINITION_FILE"])
    return config


