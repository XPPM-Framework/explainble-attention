import datetime
import os
import json
from pathlib import Path
from typing import Tuple

import pm4py
from pandas import DataFrame

from data_reader import LogReader, create_xes_file
from training import training_model
from evaluation import predict_next
from data_preprocessing import create_resource_roles, add_resource_roles, revert_activity_index_mappings

from util import create_json, get_parameter_path, get_results_path

import typer
from typing_extensions import Annotated

app = typer.Typer(pretty_exceptions_enable=False)

MY_WORKSPACE_DIR = os.getenv("MY_WORKSPACE_DIR", "../")
if not MY_WORKSPACE_DIR:
    os.environ["MY_WORKSPACE_DIR"] = str(Path(__file__).parent.resolve())

default_parameters = {
    'task': 'full_attn',  # prefix_attn
    'attention': 'full_attn',
    'experiment_name': f'experiment-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}',
    # Data
    'timeformat': '%Y-%m-%dT%H:%M:%S%z',
    'folder': str(Path(MY_WORKSPACE_DIR) / "output_files"),  # Output folder
    'log_parameters': {},

    # Model
    'imp': 1,  # keras lstm implementation 1 cpu, 2 gpu
    'lstm_act': None,  # optimization function see keras doc
    'dense_act': None,  # optimization function see keras doc
    'optim': 'Adagrad',  # optimization function see keras doc
    'norm_method': 'lognorm',  # max, lognorm
    'n_size': 5,  # n-gram size
    'model_type': 'shared_cat',  # Model types --> specialized, concatenated, shared_cat, joint, shared
    'l_size': 50,  # LSTM layer sizes
}


# Parsing logic

def parse_json_dict(value: str) -> dict:
    return json.loads(value)


@app.command()
def train(dataset_path: Path, model_path: Path,
          parameters: Annotated[dict, typer.Argument(parser=parse_json_dict)]) -> None:
    """

    :param dataset_path: Path to dataset file.
    :param model_path: The path to save the trained model to.
    :param parameters: Dictionary of parameters.
    """
    # Join with default_parameters
    final_parameters = default_parameters
    final_parameters.update(parameters)

    # Set 'file_name' value in dictionary as it is required by some other function
    final_parameters['file_name'] = str(dataset_path)
    # Set 'log_name' to filename without suffixes
    final_parameters['log_name'] = final_parameters["experiment_name"]
    final_parameters['model_path'] = str(model_path)

    timeformat = final_parameters["timeformat"]

    if dataset_path.suffix == ".gz":
        xes_path = dataset_path
        xes = pm4py.read_xes(str(xes_path))
        # df = pm4py.convert_to_dataframe(xes)
    elif dataset_path.suffix == ".csv":
        log_params = final_parameters["log_parameters"]
        xes_path = create_xes_file(dataset_path, **log_params)
        # df = xes_to_df(xes_path)
    else:
        raise Exception("Only supports zipped xes and csv files")

    log = LogReader(str(xes_path), timeformat, timeformat, one_timestamp=True)

    log_df, role_mapping = create_resource_roles(log, perform_role_mining=final_parameters.get("perform_role_mining", True))
    # Add role mapping to parameters, so they are made persisted by the train method
    final_parameters["role_mapping"] = role_mapping
    model_path.parent.mkdir(parents=True, exist_ok=True)
    create_json(final_parameters, get_parameter_path(model_path))

    model = training_model(log_df, final_parameters)

    print(f"\nSaving final model to: \"{model_path.resolve()}\"")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)


@app.command()
def explain(dataset_path: Path, model_path: Path,
            parameters: Annotated[dict, typer.Argument(parser=parse_json_dict)])\
        -> Tuple[DataFrame, DataFrame, DataFrame]:
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
    final_parameters['log_name'] = parameters["experiment_name"]
    # Set 'model_file'
    final_parameters['model_path'] = str(model_path)

    log_params = final_parameters["log_parameters"]
    timeformat = log_params["timeformat"]

    if dataset_path.suffix == ".gz":
        xes_path = dataset_path
    elif dataset_path.suffix == ".csv":
        xes_path = create_xes_file(dataset_path, **log_params)
    else:
        raise Exception("Only supports zipped xes and csv files")

    # We now still need to convert this via the LogReader to ensure the correct format
    log_df = LogReader(str(xes_path), timeformat, timeformat, one_timestamp=True).to_dataframe()

    # ensure_xes_standard_naming(log_df, final_parameters["log_parameters"])

    # Map roles according to loaded parameters
    role_mapping = final_parameters["role_mapping"]
    log_df = add_resource_roles(log_df, role_mapping)

    temporal_attentions, global_attentions, local_attentions, prefix_df = predict_next(log_df, timeformat, final_parameters)

    activity_columns = ["Prefix", "Next Activity - Ground Truth", "Next Activity - Prediction"]
    prefix_df_reverted = revert_activity_index_mappings(prefix_df, activity_columns, final_parameters["index_ac"])

    results_path = get_results_path(model_path)
    results_path.mkdir(parents=True, exist_ok=True)

    print(f"Writing results to {results_path}")

    temporal_attentions.to_csv(results_path / f"temporal_attentions.csv", index=False)
    global_attentions.to_csv(results_path / f"global_attentions.csv", index=True) if global_attentions is not None else None
    local_attentions.to_csv(results_path / f"local_attentions.csv", index=False) if local_attentions is not None else None
    prefix_df_reverted.to_csv(results_path / f"prefixes.csv", index=False)

    return global_attentions, local_attentions, prefix_df_reverted


@app.command()
def test_call(switch: str):
    if switch == "XES":
        train_dataset = Path(MY_WORKSPACE_DIR) / 'BPIC_Data/Helpdesk.xes.gz'
        test_dataset = Path(MY_WORKSPACE_DIR) / 'BPIC_Data/Helpdesk.xes.gz'
        log_params = {
            "experiment_name": "Helpesk_2017",
            "timeformat": "%Y-%m-%dT%H:%M:%S.%f",
        }
    elif switch == "CSV":
        train_dataset = Path(
            "/home/be_cracked/Dev/Uni/MA/masterthesis/framework/data/helpdesk_2017/helpdesk_2017_train_0.8.csv")
        test_dataset = Path(
            "/home/be_cracked/Dev/Uni/MA/masterthesis/framework/data/helpdesk_2017/helpdesk_2017_test_0.2.csv")
        log_params = {
            "experiment_name": "Helpesk_2017",
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

    model_path = Path(MY_WORKSPACE_DIR) / 'models/model.h5'

    # train(train_dataset, model_path, experiment_name, log_params)
    print("Training Done")

    explain(test_dataset, model_path, log_params)
    print("Explaining Done")


if __name__ == "__main__":
    #test_call("CSV")
    app()
