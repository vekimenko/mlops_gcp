# Vertex Pipelines End-to-end Samples

## Introduction

This repository provides a reference implementation of [Vertex Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/) for creating a production-ready MLOps solution on Google Cloud.

### Prerequisites

- [pyenv](https://github.com/pyenv/pyenv#installation) for managing Python versions
- [Cloud SDK](https://cloud.google.com/sdk/docs/quickstart)
- Make

### Local setup

1. Clone the repository locally
1. Install Python: `pyenv install`
1. Install pipenv and pipenv dependencies: `make setup`
1. Install pre-commit hooks: `cd pipelines && pipenv run pre-commit install`
1. Update the environment variables in `env.sh`
1. Load the environment variables in `env.sh` by running `source env.sh`

### Deploying Cloud Infrastructure

The cloud infrastructure is managed using Terraform and is defined in the [`terraform`](terraform) directory. There are three Terraform modules defined in [`terraform/modules`](terraform/modules):

- `cloudfunction` - deploys a (Pub/Sub-triggered) Cloud Function from local source code
- `scheduled_pipelines` - deploys Cloud Scheduler jobs that will trigger Vertex Pipeline runs (via the above Cloud Function)
- `vertex_deployment` - deploys Cloud infrastructure required for running Vertex Pipelines, including enabling APIs, creating buckets, service accounts, and IAM permissions.

There is a Terraform configuration for each environment (dev/test/prod) under [`terraform/envs`](terraform/envs/).

#### Deploying only the dev environment

We recommend that you set up CI/CD to deploy your environments. However, if you would prefer to deploy the dev environment manually (for example to try out the repo), you can do so as follows:

(Assuming you have an empty Google Cloud project where you are an Owner)

1. Install Terraform on your local machine. We recommend using [`tfswitch`](https://tfswitch.warrensbox.com/) to automatically choose and download an appropriate version for you (run `tfswitch` from the [`terraform/envs/dev`](terraform/envs/dev/) directory).
2. Using the `gsutil` command line tool, create a Cloud Storage bucket for the Terraform state:

```
gsutil mb -l ${VERTEX_LOCATION} -p ${VERTEX_PROJECT_ID} --pap=enforced gs://${VERTEX_PROJECT_ID}-tfstate && gsutil ubla set on gs://${VERTEX_PROJECT_ID}-tfstate
```

3. Deploy the cloud infrastructure by running the `make deploy-infra` command from the root of the repository.

#### Full deployment of dev/test/prod using CI/CD

You will need four Google Cloud projects:

- dev
- test
- prod
- admin

The Cloud Build pipelines will run in the _admin_ project, and deploy resources into the dev/test/prod projects.

Before your CI/CD pipelines can deploy the infrastructure, you will need to set up a Terraform state bucket for each environment:

```bash
gsutil mb -l <GCP region e.g. europe-west2> -p <DEV PROJECT ID> --pap=enforced gs://<DEV PROJECT ID>-tfstate && gsutil ubla set on gs://<DEV PROJECT ID>-tfstate

gsutil mb -l <GCP region e.g. europe-west2> -p <TEST PROJECT ID> --pap=enforced gs://<TEST PROJECT ID>-tfstate && gsutil ubla set on gs://<TEST PROJECT ID>-tfstate

gsutil mb -l <GCP region e.g. europe-west2> -p <PROD PROJECT ID> --pap=enforced gs://<PROD PROJECT ID>-tfstate && gsutil ubla set on gs://<PROD PROJECT ID>-tfstate
```

You will also need to manually enable the Cloud Resource Manager and Service Usage APs for your _admin_ project:

```bash
gcloud services enable cloudresourcemanager.googleapis.com --project=<ADMIN PROJECT ID>
gcloud services enable serviceusage.googleapis.com --project=<ADMIN PROJECT ID>
```

Now that you have created a Terraform state bucket for each environment, you can set up the CI/CD pipelines. You can find instructions for this [here](cloudbuild/README.md).

#### Tearing down infrastructure

To tear down the infrastructure you have created with Terraform, run `make destroy-infra`.

### Example ML pipelines

This repository contains example ML training and prediction pipelines for two popular frameworks (XGBoost/sklearn and Tensorflow) using the popular [Chicago Taxi Dataset](https://console.cloud.google.com/marketplace/details/city-of-chicago-public-data/chicago-taxi-trips). The details of these can be found in the [separate README](pipelines/README.md).

#### Pre-requisites

Before you can run these example pipelines successfully there are a few additional things you will need to deploy (they have not been included in the Terraform code as they are specific to these pipelines)

1. Create a new BigQuery dataset for the Chicago Taxi data:

```
bq --location=${VERTEX_LOCATION} mk --dataset "${VERTEX_PROJECT_ID}:chicago_taxi_trips"
```

2. Create a new BigQuery dataset for data processing during the pipelines:

```
bq --location=${VERTEX_LOCATION} mk --dataset "${VERTEX_PROJECT_ID}:preprocessing"
```

3. Set up a BigQuery transfer job to mirror the Chicago Taxi dataset to your project

```
bq mk --transfer_config \
  --project_id=${VERTEX_PROJECT_ID} \
  --data_source="cross_region_copy" \
  --target_dataset="chicago_taxi_trips" \
  --display_name="Chicago taxi trip mirror" \
  --params='{"source_dataset_id":"'"chicago_taxi_trips"'","source_project_id":"'"bigquery-public-data"'"}'
```

#### Running Pipelines

You can run the XGBoost training pipeline (for example) with:

```bash
make run PIPELINE_TEMPLATE=xgboost pipeline=training
```

Alternatively, add the environment variable `PIPELINE_TEMPLATE=xgboost` and/or `pipeline=training` to `env.sh`, then:

```bash
make run pipeline=<training|prediction>
```

This will execute the pipeline using the chosen template on Vertex AI, namely it will:

1. Compile the pipeline using the Kubeflow Pipelines SDK
1. Copy the `assets` folders to Cloud Storage
1. Trigger the pipeline with the help of `pipelines/trigger/main.py`

#### Pipeline input parameters

The ML pipelines have input parameters. As you can see in the pipeline definition files (`pipelines/pipelines/<xgboost|tensorflow>/<training|prediction>/pipeline.py`), they have default values, and some of these default values are derived from environment variables (which in turn are defined in `env.sh`).

When triggering ad hoc runs in your dev/sandbox environment, or when running the E2E tests in CI, these default values are used. For the test and production deployments, the pipeline parameters are defined in the Terraform code for the Cloud Scheduler jobs (`terraform/envs/<test|prod>/variables.auto.tfvars`).

### Assets

In each pipeline folder, there is an `assets` directory (`pipelines/pipelines/<xgboost|tensorflow>/<training|prediction>/assets/`). 
This can be used for any additional files that may be needed during execution of the pipelines. 
This directory is rsync'd to Google Cloud Storage when running a pipeline in the sandbox environment or as part of the CD pipeline (see [CI/CD setup](cloudbuild/README.md)).

## Testing

Unit tests and end-to-end (E2E) pipeline tests are performed using [pytest](https://docs.pytest.org). 
The unit tests for custom KFP components are run on each pull request, and the E2E tests are run on merge to the main branch. To run them on your local machine:

```
make setup-all-components
make test-all-components
```

Alternatively, only setup and install one of the components groups by running:
```
make setup-components GROUP=vertex-components
make test-components GROUP=vertex-components
```

To run end-to-end tests of a single pipeline, you can use:

```
make e2e-tests pipeline=<training|prediction> [ enable_caching=<true|false> ] [ sync_assets=<true|false> ]
```

There are also unit tests for the pipeline triggering code. 
This is not run as part of a CI/CD pipeline, as we don't expect this to be changed for each use case. To run them on your local machine:

```
make test-trigger
```

## Customize pipelines

### Update existing pipelines

See existing [XGBoost](pipelines/src/pipelines/xgboost) and [Tensorflow](pipelines/src/pipelines/tensorflow) pipelines as part of this template.
Update `PIPELINE_TEMPLATE` to `xgboost` or `tensorflow` in [env.sh](env.sh.example) to specify whether to run the XGBoost pipelines or TensorFlow pipelines. 
Make changes to the ML pipelines and their associated tests.
Refer to the [contribution instructions](CONTRIBUTING.md) for more information on committing changes. 

### Scheduling pipelines

Terraform is used to deploy Cloud Scheduler jobs that trigger the Vertex Pipeline runs. This is done by the CI/CD pipelines (see section below on CI/CD).

To schedule pipelines into an environment, you will need to provide the `cloud_schedulers_config` variable to the Terraform configuration for the relevant environment. You can find an example of this configuration in [`terraform/modules/scheduled_pipelines/scheduled_jobs.auto.tfvars.example`](terraform/modules/scheduled_pipelines/scheduled_jobs.auto.tfvars.example). Copy this example file into the relevant directory for your environment (e.g. `terraform/envs/dev` for the dev environment) and remove the `.example` suffix. Adjust the configuration file as appropriate.

### CI/CD

There are five CI/CD pipelines located under the [cloudbuild](cloudbuild) directory:

1. `pr-checks.yaml` - runs pre-commit checks and unit tests on the custom KFP components, and checks that the ML pipelines (training and prediction) can compile.
2. `e2e-test.yaml` - copies the "assets" folders to the chosen GCS destination (versioned by git commit hash - see below) and runs end-to-end tests of the training and prediction pipeline.
3. `release.yaml` - Compiles the training and prediction pipelines, and copies the compiled pipelines, along with their respective `assets` directories, to Google Cloud Storage in the build / CI/CD environment. The Google Cloud Storage destination is namespaced using the git tag (see below). Following this, the E2E tests are run on the new compiled pipelines.

Below is a diagram of how the files are published in each environment in the `e2e-test.yaml` and `release.yaml` pipelines:

```
. <-- GCS directory set by _PIPELINE_PUBLISH_GCS_PATH
└── TAG_NAME or GIT COMMIT HASH <-- Git tag used for the release (release.yaml) OR git commit hash (e2e-test.yaml)
    ├── prediction
    │   ├── assets
    │   │   └── some_useful_file.json
    │   └── prediction.json   <-- compiled prediction pipeline
    └── training
        ├── assets
        │   └── training_task.py
        └── training.json   <-- compiled training pipeline
```

4. `terraform-plan.yaml` - Checks the Terraform configuration under `terraform/envs/<env>` (e.g. `terraform/envs/test`), and produces a summary of any proposed changes that will be applied on merge to the main branch.
5. `terraform-apply.yaml` - Applies the Terraform configuration under `terraform/envs/<env>` (e.g. `terraform/envs/test`).

