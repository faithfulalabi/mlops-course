from distutils.command.config import config
from unicodedata import name
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from prefect import flow, task


from prefect import get_run_logger

# print = get_run_print()


@task
def read_data(path):
    df = pd.read_parquet(path)
    return df


def get_path(date: datetime = None):
    if date is None:
        date = datetime.now().date()
        # use the data from 2 months back as training path
        training_path = "./data/fhv_tripdata_2021-01.parquet"

        # use the data from previous month as val data
        val_path = "./data/fhv_tripdata_2021-01.parquet"
    else:
        if type(date) == str:
            date = datetime.strptime(date, "%Y-%m-%d").date()

            train_date = date - relativedelta(months=2)
            train_date_str = train_date.strftime("%Y-%m")
            val_date = date - relativedelta(months=1)
            val_date_str = val_date.strftime("%Y-%m")
            training_path = f"./data/fhv_tripdata_{train_date_str}.parquet"
            val_path = f"./data/fhv_tripdata_{val_date_str}.parquet"
            # /Users/faithful/Desktop/Data_Sci_Projects/mlops-course/03-orchestration/
    return training_path, val_path


@task
def prepare_features(df, categorical, train=True):
    df["duration"] = df.dropOff_datetime - df.pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        print(f"The mean duration of training is {mean_duration}")
    else:
        print(f"The mean duration of validation is {mean_duration}")

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")
    return df


@task
def train_model(df, categorical):
    logger = get_run_logger()
    train_dicts = df[categorical].to_dict(orient="records")
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df.duration.values

    print(f"The shape of X_train is {X_train.shape}")
    print(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv


@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    val_dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return


@flow(name="MLOPS COURSE Q4")
def main(date="2021-08-15"):
    logger = get_run_logger()
    categorical = ["PUlocationID", "DOlocationID"]
    train_path, val_path = get_path(date=date)
    logger.info(f"train_path is {train_path}")
    logger.info("train_path is {val_path}")

    df_train = read_data(train_path)
    logger.info(df_train)
    df_train_processed = prepare_features(df_train, categorical)
    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)
    with open(f"model-{date}.bin", "wb") as f_out:
        pickle.dump(lr, f_out)

    with open(f"dv-{date}.bin", "wb") as f_out:
        pickle.dump(dv, f_out)


from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.orion.schemas.schedules import IntervalSchedule
from prefect.flow_runners import SubprocessFlowRunner


DeploymentSpec(
    flow=main,
    name="cron-schedule-deployment_3.0",
    # flow_location="/Users/faithful/Desktop/Data_Sci_Projects/mlops-course/03-orchestration/homework.py",
    schedule=CronSchedule(cron="0 9 15 * *", timezone="America/Chicago"),
    flow_runner=SubprocessFlowRunner(),  # specify it only runs on your local machine
    tags=["mlops-course"],
)
main()
