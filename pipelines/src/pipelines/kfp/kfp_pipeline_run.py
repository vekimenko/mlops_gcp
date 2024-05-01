from kfp.v2.dsl import component, Artifact, Dataset, Input, Metrics, ClassificationMetrics, Model, Output
from typing import NamedTuple
from kfp.v2.dsl import pipeline
from kfp.v2 import compiler
from kfp.v2.dsl import Condition
import sys

# sys.path.append("../..")
# from src.common import utils
from . import utils
from google.cloud import aiplatform as aip
import logging


@component(
    base_image="europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
    packages_to_install=["pandas-gbq"]
)
def data_download(
        train_query: str,
        test_query: str,
        project: str,
        dataset_train: Output[Dataset],
        dataset_test: Output[Dataset],
):
    import pandas as pd
    import logging

    def preprocess(query):
        df = pd.read_gbq(query, project_id=project)
        df['user_agent'] = df['user_agent'].astype('string')
        df = df[df['status'] != 'TBD']
        df['target'] = 0
        df['target'] = df['target'].mask(df['status'] == 1, 1)
        df = df.drop(columns='status')
        df_class_0 = df[df['target'] == 0]
        df_class_1 = df[df['target'] == 1]
        class_count_1, class_count_0 = df['target'].value_counts()
        logging.info('y counts')
        logging.info(class_count_1)
        logging.info(class_count_0)
        df_class_1_under = df_class_1.sample(class_count_0, random_state=42)
        df_test_under = pd.concat([df_class_1_under, df_class_0], axis=0)
        return df_test_under

    _dataset_train = preprocess(train_query)
    _dataset_test = preprocess(test_query)

    _dataset_train.to_csv(dataset_train.path, index=False)
    _dataset_test.to_csv(dataset_test.path, index=False)


@component(
    base_image="europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
)
def model_train(
        dataset: Input[Dataset],
        model: Output[Artifact],
):
    import pandas as pd
    import pickle
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss
    import logging

    data = pd.read_csv(dataset.path)
    data.info()
    x_train = data[['user_agent']].values.tolist()
    y_train = data["target"].values.tolist()
    
    vect = TfidfVectorizer(lowercase=False, ngram_range=(1, 3))
    
    preprocess = ColumnTransformer(
        [('tfidf', vect, 0)])
        
    classifier = RandomForestClassifier(
        random_state=42,
        bootstrap=True,
        n_jobs=-1,
        max_depth=20,
        max_features='sqrt',
        min_samples_leaf=2,
        min_samples_split=4,
        n_estimators=50
    )
        
    pipeline = Pipeline([
        ('tfidf', preprocess),
        ('rf', classifier)
    ])

    pipeline.fit(x_train, y_train)

    def model_calibration(model, X_train, y_train):
        """
        Calibrate the provided model using the training data.
        """
        calibrator = CalibratedClassifierCV(model, cv='prefit', method='isotonic')
        calibrator.fit(X_train, y_train)
        return calibrator

    model_calibrated = model_calibration(pipeline, x_train, y_train)

    model.metadata["framework"] = "scikit-learn"
    model.metadata["containerSpec"] = {
        "imageUri": "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
    }

    file_name = model.path + "/model.pkl"
    import pathlib

    pathlib.Path(model.path).mkdir()
    with open(file_name, "wb") as file:
        pickle.dump(model_calibrated, file)


@component(
    base_image="europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
)
def model_evaluate(
        test_set: Input[Dataset],
        model: Input[Model],
        threshold_dict_str: str,
        metrics: Output[ClassificationMetrics],
        kpi: Output[Metrics],
) -> NamedTuple("output", [("deploy", str)]):
    import pandas as pd
    import pickle
    import json
    from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score

    def threshold_check(val1, val2):
        cond = "false"
        if val1 >= val2:
            cond = "true"
        return cond

    data = pd.read_csv(test_set.path)
    file_name = model.path + "/model.pkl"
    with open(file_name, "rb") as file:
        model_pipeline = pickle.load(file)

    X = data[['user_agent']].values.tolist()
    y = data["target"].values.tolist()
    y_pred = model_pipeline.predict(X)

    y_scores = model_pipeline.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true=y, y_score=y_scores, pos_label=True)
    metrics.log_roc_curve(fpr.tolist(), tpr.tolist(), thresholds.tolist())

    metrics.log_confusion_matrix(
        ["False", "True"],
        confusion_matrix(y, y_pred).tolist(),
    )

    _accuracy = model_pipeline.score(X, y)
    accuracy = float(_accuracy)
    kpi.log_metric("accuracy", accuracy)

    threshold_dict = json.loads(threshold_dict_str)
    deploy = threshold_check(accuracy, int(threshold_dict['roc']))
    return (deploy,)


@component(
    base_image="europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
)
def model_deploy(
        dest_path: str,
        model: Input[Model]
) -> NamedTuple("output", [("path", str)]):
    from google.cloud import storage
    import logging
    storage_client = storage.Client()
    src_path = str(model.path).replace("/gcs/", "")
    source_bucket_name, source_blob_name = src_path.split("/", 1)
    source_bucket = storage_client.bucket(source_bucket_name)
    logging.info(source_bucket_name)
    dest_bucket_name, dest_blob_name = dest_path.replace("gs://", "").split("/", 1)
    destination_bucket = storage_client.bucket(dest_bucket_name)
    logging.info(dest_bucket_name)
    blobs = storage_client.list_blobs(source_bucket_name, prefix=source_blob_name, delimiter=None)
    for blob in blobs:
        if not blob.name.endswith("/"):
            new_destination_blob_name = dest_blob_name.rstrip("/") + "/" + blob.name[len(source_blob_name):].lstrip("/")
            logging.info(new_destination_blob_name)
            source_bucket.copy_blob(blob, destination_bucket, new_destination_blob_name)
    return (dest_path,)


def _create_pipeline(
        name: str,
        pipeline_root: str,
        train_query: str,
        test_query: str,
        project: str,
        threshold_dict_str: str,
        dest_path: str
):
    @pipeline(name=name, pipeline_root=pipeline_root)
    def _pipeline():
        data_op = data_download(
            train_query=train_query,
            test_query=test_query,
            project=project
        )

        model_train_op = model_train(
            dataset=data_op.outputs["dataset_train"]
        )

        model_evaluate_op = model_evaluate(
            test_set=data_op.outputs["dataset_test"],
            model=model_train_op.outputs["model"],
            threshold_dict_str=threshold_dict_str
        )

        with Condition(
                model_evaluate_op.outputs["deploy"] == "true",
                name="deploy-UserAgentTensorflowHub",
        ):
            model_deploy_op = model_deploy(
                dest_path=dest_path,
                model=model_train_op.outputs['model'])

    return _pipeline


def _compile(package_path, config):
    _pipe = _create_pipeline(
        name=config["DATASET_DISPLAY_NAME"],
        pipeline_root=config["PIPELINE_ROOT"],
        train_query=config["TRAIN_QUERY"],
        test_query=config["TEST_QUERY"],
        project=config["PROJECT"],
        threshold_dict_str='{"roc":0.1}',
        dest_path=config["DEST_PATH"]
    )
    compiler.Compiler().compile(
        pipeline_func=_pipe,
        package_path=package_path
    )


def main():
    print("start")
    config = utils.load_config()
    print("config")
    logging.info("config from logging")
    logging.info(str(config))
    _compile("pipeline.json", config)
    aip.init(project=config["PROJECT"], staging_bucket="gs://mlops-gcp/tmp", location=config["REGION"])
    job = aip.PipelineJob(
        display_name="UserAgent",
        template_path="pipeline.json",
        pipeline_root=config["PIPELINE_ROOT"],
        enable_caching=True
    )
    job.run(service_account=config["SERVICE_ACCOUNT"])


if __name__ == '__main__':
    main()