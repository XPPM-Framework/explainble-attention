import asyncio
import csv
import datetime
import json
import os
import uuid
from logging import Logger
from pathlib import Path
from sys import stdout
from typing import Union
import numpy as np


def folder_id():
    return datetime.datetime.today().strftime('%Y%m%d_%H%M%S%f')


# generate unique bimp element ids
def gen_id():
    return "qbp_" + str(uuid.uuid4())


def print_performed_task(text):
    stdout.write("\r%s" % text + "...      ")
    stdout.flush()


def print_done_task():
    stdout.write("[DONE]")
    stdout.flush()
    stdout.write("\n")


def file_size(path_file):
    size = 0
    file_exist = os.path.exists(path_file)
    if file_exist:
        size = len(open(path_file).readlines())
    return size


# printing formated float
def ffloat(num, dec):
    return float("{0:.2f}".format(np.round(num, decimals=dec)))


def get_parameter_path(model_path: Path) -> Path:
    """
    Get the path to the model-parameter.json file next to the model.
    """
    model_path = Path(model_path)
    return model_path.with_name(f"{model_path.stem}-parameters.json")


def get_results_path(model_path: Path) -> Path:
    """
    Get the path to folder to which to write the results of a prediction/explanation run.
    """
    return Path(model_path).parent


# transform a string into date object
# def get_time_obj(date, timeformat):
#    date_modified = datetime.datetime.strptime(date,timeformat)
#    return date_modified


# print debuging csv file
def create_csv_file(index, output_file, mode='w'):
    with open(output_file, mode) as f:
        for element in index:
            w = csv.DictWriter(f, element.keys())
            w.writerow(element)
        f.close()


def create_csv_file_header(index, output_file, mode='w'):
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, mode, newline='') as f:
        fieldnames = index[0].keys()
        w = csv.DictWriter(f, fieldnames)
        w.writeheader()
        for element in index:
            w.writerow(element)
        f.close()


def create_json(dictionary, output_file):
    with open(output_file, 'w') as f:
        f.write(json.dumps(dictionary, indent=4))


def read_json(file_path: Union[Path, str]) -> dict:
    with open(file_path, 'r') as f:
        return json.load(f)


# rounding lists values preserving the sum values
def round_preserve(l, expected_sum):
    actual_sum = sum(l)
    difference = round(expected_sum - actual_sum, 2)
    if difference > 0.00:
        idx = l.index(min(l))
    else:
        idx = l.index(max(l))
    l[idx] += difference
    return l


def plot_history(plt, figure_name, path: Union[Path, str], save_fig=True):
    fig_name = figure_name + ".png"
    full_path = Path(path) / fig_name
    if save_fig:
        plt.savefig(full_path, dpi=300)
