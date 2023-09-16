import json
from pathlib import Path
from typing import Union

import pandas as pd
import pm4py

from data_reader import LogReader
from training import training_model
from evaluation import predict_next
from data_preparation import create_resource_roles, add_resource_roles

from util import create_json, get_parameter_path

import os

MY_WORKSPACE_DIR = os.getenv("MY_WORKSPACE_DIR", "../")

default_parameters = {
    'task': 'prefix_attn',
    # Data
    'timeformat': '%Y-%m-%dT%H:%M:%S.%f',
    'folder': str(Path(MY_WORKSPACE_DIR) / "output_files"),  # Output folder

    # Model
    'imp': 1,  # keras lstm implementation 1 cpu, 2 gpu
    'lstm_act': None,  # optimization function see keras doc
    'dense_act': None,  # optimization function see keras doc
    'optim': 'Adagrad',  # optimization function see keras doc
    'norm_method': 'lognorm',  # max, lognorm
    'n_size': 15,  # n-gram size
    'model_type': 'shared_cat',  # Model types --> specialized, concatenated, shared_cat, joint, shared
    'l_size': 50,  # LSTM layer sizes
}


def train(dataset_path: Path, model_path: Path, experiment_name: str, **parameters: dict) -> None:
    """

    :param dataset_path: Path to dataset file.
    :param model_path: The path to save the trained model to.
    :param experiment_name: Name of the experiment that is being run. Is used in saved data file paths/names.
    :param parameters: Dictionary of parameters.
    """
    # Join with default_parameters
    final_parameters = default_parameters
    final_parameters.update(parameters)

    # Set 'file_name' value in dictionary as it is required by some other function
    final_parameters['file_name'] = str(dataset_path)
    # Set 'log_name' to filename without suffixes
    final_parameters['log_name'] = experiment_name
    final_parameters['model_path'] = str(model_path)

    timeformat = final_parameters["timeformat"]

    if dataset_path.suffix == ".gz":
        xes_path = dataset_path
        xes = pm4py.read_xes(str(xes_path))
        df = pm4py.convert_to_dataframe(xes)
    elif dataset_path.suffix == ".csv":
        log_params = final_parameters["log_parameters"]
        xes_path = create_xes_file(dataset_path, **log_params)
        # df = xes_to_df(xes_path)
    else:
        raise Exception("Only supports zipped xes and csv files")

    log = LogReader(str(xes_path), timeformat, timeformat, one_timestamp=True)

    log_df, role_mapping = create_resource_roles(log)
    # Add role mapping to parameters, so they are made persisted by the train method
    final_parameters["role_mapping"] = role_mapping
    # TODO: Persist role_mapping next to model file with same name as json
    model_path.parent.mkdir(parents=True, exist_ok=True)
    create_json(final_parameters, get_parameter_path(model_path))

    model = training_model(log_df, final_parameters)

    print(f"\nSaving final model to: \"{model_path.resolve()}\"")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)


def explain(dataset_path: Path, model_path: Path, experiment_name: str, **parameters: dict) -> None:
    # Load parameters
    with open(get_parameter_path(model_path)) as file:
        loaded_parameters = json.load(file)

    # Join with default_parameters
    final_parameters = default_parameters
    final_parameters.update(loaded_parameters)
    final_parameters.update(parameters)
    # Set 'file_name' value in dictionary as it is required by some other function
    final_parameters['file_name'] = str(dataset_path)
    # Set 'log_name' to filename without suffixes
    final_parameters['log_name'] = experiment_name
    # Set 'model_file'
    final_parameters['model_path'] = str(model_path)

    timeformat = final_parameters["timeformat"]

    if dataset_path.suffix == ".gz":
        xes_path = dataset_path
        xes = pm4py.read_xes(str(xes_path))
    elif dataset_path.suffix == ".csv":
        log_params = final_parameters["log_parameters"]
        xes_path = create_xes_file(dataset_path, **log_params)
    else:
        raise Exception("Only supports zipped xes and csv files")

    # We now still need to convert this via the LogReader to ensure the correct format
    log_df = LogReader(str(xes_path), timeformat, timeformat, one_timestamp=True).data

    log_df = pd.read_csv(dataset_path)
    ensure_xes_standard_naming(log_df, final_parameters["log_parameters"])

    # Map roles according to loaded parameters
    role_mapping = final_parameters["role_mapping"]
    log_df = add_resource_roles(log_df, role_mapping, resource_column=pm4py.util.xes_constants.DEFAULT_RESOURCE_KEY)

    predict_next(log_df, timeformat, final_parameters)


def create_xes_file(csv_path: Path, *, xes_path: Union[Path, str] = None, **log_parameter) -> Path:
    """
    Load a csv file, parse it to a xes file with pm4py so that it will contain all required columns,
     then write the xes file next to the csv.
    \n Adds 'lifecycle:transition' column if necessary.
    \n Maps dataframe column names to the pm4py defaults.
    :param csv_path: Path to a csv file of an event log.
    :param xes_path: The path to save the xes file to. Defaults to csv_path with changed suffix.
    :param log_parameter: Specifically mappings for names of certain column keys.
    :return: The path of the xes file next to the
    """
    df = pd.read_csv(csv_path)
    import pm4py

    df = ensure_xes_standard_naming(df, log_parameter)

    df = pm4py.format_dataframe(df)
    event_log = pm4py.convert_to_event_log(df)
    if not xes_path:
        xes_path = csv_path.with_suffix(".xes")
    pm4py.write_xes(event_log, str(xes_path))

    return xes_path


def ensure_xes_standard_naming(log_df: pd.DataFrame, log_parameter) -> pd.DataFrame:
    """
    Ensures that naming conventions of the XES standard are adhered to.
    \nThis includes adding a "life_cycle:transition" column should it not exist. This defaults to complete.
    :param log_df:
    :param log_parameter: Specifically mappings for names of certain column keys.
    :return:
    """
    import pm4py.util.xes_constants as xes_const

    if xes_const.DEFAULT_TRANSITION_KEY not in log_df.columns:  # "lifecycle:transition"
        log_df[xes_const.DEFAULT_TRANSITION_KEY] = "complete"

    # Construct rename mapping to fit the pm4py xes standards from the log parameters
    xes_column_remap = {
        log_parameter["case_id_key"]: xes_const.DEFAULT_NAME_KEY,  # "concept:name"
        log_parameter["activity_key"]: "case:" + xes_const.DEFAULT_NAME_KEY,  # "case:concept:name"
        log_parameter["timestamp_key"]: xes_const.DEFAULT_TIMESTAMP_KEY,  # "time:timestamp"
        log_parameter["resource_key"]: "org:resource",
    }
    log_df.rename(xes_column_remap, axis=1, inplace=True)

    return log_df


# TODO: Remove?
def xes_to_df(xes_path: Path, timeformat: str = "%Y-%m-%dT%H:%M:%S%z") -> pd.DataFrame:
    from data_reader import read_resource_pool

    log = LogReader(xes_path, timeformat, timeformat, one_timestamp=True)
    _, resource_table = read_resource_pool(log, sim_percentage=0.50)
    log_df = pd.DataFrame.from_records(log.data)
    if len(resource_table) > 0:
        # Role discovery
        log_df_resources = pd.DataFrame.from_records(resource_table)
        log_df_resources = log_df_resources.rename(index=str, columns={"resource": "user"})
        # Dataframe creation
        log_df = log_df.merge(log_df_resources, on='user', how='left')
    log_df = log_df[log_df.task != 'Start']
    log_df = log_df[log_df.task != 'End']
    log_df = log_df.reset_index(drop=True)

    return log_df


def main(switch: str):
    if switch == "XES":
        train_dataset = Path(MY_WORKSPACE_DIR) / 'BPIC_Data/Helpdesk.xes.gz'
        test_dataset = Path(MY_WORKSPACE_DIR) / 'BPIC_Data/Helpdesk.xes.gz'
        log_params = {
            "timeformat": "%Y-%m-%dT%H:%M:%S.%f",
        }
    elif switch == "CSV":
        train_dataset = Path(
            "/home/be_cracked/Dev/Uni/MA/masterthesis/framework/data/helpdesk_2017/helpdesk_2017_train_0.8.csv")
        test_dataset = Path(
            "/home/be_cracked/Dev/Uni/MA/masterthesis/framework/data/helpdesk_2017/helpdesk_2017_test_0.2.csv")
        log_params = {
            "timeformat": "%Y-%m-%dT%H:%M:%S%z",
            "log_parameters": {
                "case_id_key": "Case ID",
                "activity_key": "Activity",
                "timestamp_key": "Complete Timestamp",
                "timest_format": "%Y-%m-%dT%H:%M:%S.%f",
                "resource_key": "Resource",
            }
        }
    else:
        raise Exception()

    experiment_name = "Helpesk_2017"
    model_path = Path(MY_WORKSPACE_DIR) / 'models/model.h5'

    #train(train_dataset, model_path, experiment_name, **log_params)
    print("Training Done")

    explain(test_dataset, model_path, experiment_name, **log_params)
    print("Explaining Done")


if __name__ == "__main__":
    main("CSV")  # XES or CSV
