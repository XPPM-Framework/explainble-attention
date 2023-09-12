from pathlib import Path

from data_reader import LogReader
from training import training_model

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


def catch_parameter(opt):
    """Change the captured parameters names"""
    switch = {'-h': 'help', '-i': 'imp', '-l': 'lstm_act',
              '-d': 'dense_act', '-n': 'norm_method', '-f': 'folder',
              '-m': 'model_file', '-t': 'model_type', '-a': 'activity',
              '-e': 'file_name', '-b': 'n_size', '-c': 'l_size', '-o': 'optim'}
    try:
        return switch[opt]
    except:
        raise Exception('Invalid option ' + opt)


# --setup--
def run():
    timeformat = '%Y-%m-%dT%H:%M:%S.%f'
    parameters = dict()
    #   Parameters setting manual fixed or catched by console for batch operations

    parameters['folder'] = str(Path(MY_WORKSPACE_DIR) / "output_files")
    #       Specific model training parameters
    parameters['imp'] = 1  # keras lstm implementation 1 cpu, 2 gpu
    parameters['lstm_act'] = None  # optimization function see keras doc
    parameters['dense_act'] = None  # optimization function see keras doc
    parameters['optim'] = 'Adagrad'  # optimization function see keras doc
    parameters['norm_method'] = 'lognorm'  # max, lognorm
    # Model types --> specialized, concatenated, shared_cat, joint, shared
    parameters['model_type'] = 'shared_cat'
    parameters['l_size'] = 50  # LSTM layer sizes
    #       Generation parameters
    # parameters['model_file'] = 'model_rd_100 Nadam_02-0.90.h5'
    parameters['n_size'] = 15  # n-gram size

    parameters['log_name'] = 'helpdesk'  # Sort of an "experiment name"?

    parameters['task'] = 'prefix_attn'

    parameters['file_name'] = str(Path(MY_WORKSPACE_DIR) / 'BPIC_Data/Helpdesk.xes.gz')  # 'BPI_2012_W_complete.xes.gz'

    log = LogReader(parameters['file_name'], timeformat, timeformat, one_timestamp=True)

    training_model(log, timeformat, parameters)


def train(dataset_path: Path, model_path: Path, **parameters: dict) -> None:
    """

    :param dataset_path: Path to dataset file.
    :param parameters: Dictionary of parameters.
    :param model_path: The path to save the trained model to.
    """
    # Join with default_parameters
    final_parameters = default_parameters
    final_parameters.update(parameters)

    # Set 'file_name' value in dictionary as it is required by some other function
    final_parameters['file_name'] = str(dataset_path)
    # Set 'log_name' to filename without suffixes
    final_parameters['log_name'] = dataset_path.with_suffix("").with_suffix("").name

    timeformat = final_parameters["timeformat"]
    log = LogReader(dataset_path, timeformat, timeformat, one_timestamp=True)

    model = training_model(log, timeformat, final_parameters)

    print(f"\nSaving final model to: \"{model_path.resolve()}\"")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)


def main():
    dataset = Path(MY_WORKSPACE_DIR) / 'BPIC_Data/Helpdesk.xes.gz'
    params = {}
    model = Path(MY_WORKSPACE_DIR) / 'models/model.h5'
    train(dataset, model, **params)


if __name__ == "__main__":
    main()
