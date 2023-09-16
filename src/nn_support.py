# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 16:53:16 2018
This module contains support functions specifically created to manipulate
Event logs in pandas dataframe format
@author: Manuel Camargo
"""
import itertools
import math

import numpy as np
import pandas as pd
import keras.utils as ku
from nltk.util import ngrams


# =============================================================================
# Split an event log dataframe to peform split-validation
# =============================================================================
def split_train_test(df, percentage):
    cases = df.caseid.unique()
    num_test_cases = int(np.round(len(cases) * percentage))
    test_cases = cases[:num_test_cases]
    train_cases = cases[num_test_cases:]
    df_train, df_test = pd.DataFrame(), pd.DataFrame()

    # I hate this more than you
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)

        for case in train_cases:
            df_train = df_train.append(df[df.caseid == case])
        df_train = df_train.sort_values('start_timestamp', ascending=True).reset_index(drop=True)

        for case in test_cases:
            df_test = df_test.append(df[df.caseid == case])
        df_test = df_test.sort_values('start_timestamp', ascending=True).reset_index(drop=True)

    return df_train, df_test


# =============================================================================
# Reduce the loops of a trace joining contiguous activities
# exectuted by the same resource
# =============================================================================
def reduce_loops(df):
    df_group = df.groupby('caseid')
    reduced = list()
    for name, group in df_group:
        temp_trace = list()
        group = group.sort_values('start_timestamp', ascending=True).reset_index(drop=True)
        temp_trace.append(dict(caseid=name,
                               task=group.iloc[0].task,
                               user=group.iloc[0].user,
                               start_timestamp=group.iloc[0].start_timestamp,
                               end_timestamp=group.iloc[0].end_timestamp,
                               role=group.iloc[0].role))
        for i in range(1, len(group)):
            if group.iloc[i].task == temp_trace[-1]['task'] and group.iloc[i].user == temp_trace[-1]['user']:
                temp_trace[-1]['end_timestamp'] = group.iloc[i].end_timestamp
            else:
                temp_trace.append(dict(caseid=name,
                                       task=group.iloc[i].task,
                                       user=group.iloc[i].user,
                                       start_timestamp=group.iloc[i].start_timestamp,
                                       end_timestamp=group.iloc[i].end_timestamp,
                                       role=group.iloc[i].role))
        reduced.extend(temp_trace)
    return pd.DataFrame.from_records(reduced)


# =============================================================================
# Calculate duration and time between activities
# =============================================================================
def calculate_times(df):
    # Duration
    get_seconds = lambda x: x.seconds
    df['dur'] = (df.end_timestamp - df.start_timestamp).apply(get_seconds)
    # Time between activities per trace
    df['tbtw'] = 0
    # Multitasking time
    cases = df.caseid.unique()
    for case in cases:
        trace = df[df.caseid == case].sort_values('start_timestamp', ascending=True)
        for i in range(1, len(trace)):
            row_num = trace.iloc[i].name
            tbtw = (trace.iloc[i].start_timestamp - trace.iloc[i - 1].end_timestamp).seconds
            df.iloc[row_num, df.columns.get_loc('tbtw')] = tbtw
    return df, cases


# =============================================================================
# Standardization
# =============================================================================

def max_min_std(df, serie):
    max_value, min_value = np.max(df[serie]), np.min(df[serie])
    std = lambda x: (x[serie] - min_value) / (max_value - min_value)
    df[serie + '_norm'] = df.apply(std, axis=1)
    return df, max_value, min_value


def max_std(df, serie):
    max_value, min_value = np.max(df[serie]), np.min(df[serie])
    std = lambda x: x[serie] / max_value
    df[serie + '_norm'] = df.apply(std, axis=1)
    return df, max_value, min_value


def max_min_de_std(val, max_value, min_value):
    true_value = (val * (max_value - min_value)) + min_value
    return true_value


def max_de_std(val, max_value, min_value):
    true_value = val * max_value
    return true_value


def add_calculated_features(log_df, ac_index, rl_index):
    """Appends the indexes and relative time to the dataframe.
    Args:
        log_df: dataframe.
        ac_index (dict): index of activities.
        rl_index (dict): index of roles.
    Returns:
        Dataframe: The dataframe with the calculated features added.
    """
    ac_idx = lambda x: ac_index[x['task']]
    log_df['ac_index'] = log_df.apply(ac_idx, axis=1)

    rl_idx = lambda x: rl_index[x['role']]
    log_df['rl_index'] = log_df.apply(rl_idx, axis=1)

    log_df['tbtw'] = 0
    log_df['tbtw_norm'] = 0

    log_dict = log_df.to_dict('records')

    log_dict = sorted(log_dict, key=lambda x: (x['caseid'], x['end_timestamp']))
    for _, group in itertools.groupby(log_dict, key=lambda x: x['caseid']):
        trace = list(group)
        for i, _ in enumerate(trace):
            if i != 0:
                trace[i]['tbtw'] = (trace[i]['end_timestamp'] -
                                    trace[i - 1]['end_timestamp']).total_seconds()

    return pd.DataFrame.from_records(log_dict)


def vectorization(log_df: pd.DataFrame, ac_index: dict, rl_index: dict, norm_method: str, n_size: int, **kwargs):
    """Example function with types documented in the docstring.
    Args:
        log_df (dataframe): event log data.
        ac_index (dict): index of activities.
        rl_index (dict): index of roles.
        norm_method (str): What normalization method to use. 'max' or 'lognorm'.
        n_size (int): The n-gram size
    Returns:
        dict: Dictionary that contains all the LSTM inputs.
    """
    if norm_method == 'max':
        mean_tbtw = np.mean(log_df.tbtw)
        std_tbtw = np.std(log_df.tbtw)
        norm = lambda x: (x['tbtw'] - mean_tbtw) / std_tbtw
        log_df['tbtw_norm'] = log_df.apply(norm, axis=1)
        log_df = reformat_events(log_df, ac_index, rl_index)
    elif norm_method == 'lognorm':
        logit = lambda x: math.log1p(x['tbtw'])
        log_df['tbtw_log'] = log_df.apply(logit, axis=1)
        mean_tbtw = np.mean(log_df.tbtw_log)
        std_tbtw = np.std(log_df.tbtw_log)
        norm = lambda x: (x['tbtw_log'] - mean_tbtw) / std_tbtw
        log_df['tbtw_norm'] = log_df.apply(norm, axis=1)
        log_df = reformat_events(log_df, ac_index, rl_index)
    else:
        raise ValueError(f"Invalid parameter 'norm_method' {norm_method}")
    args = dict()

    vec = {'prefixes': dict(), 'next_evt': dict(), 'mean_tbtw': mean_tbtw, 'std_tbtw': std_tbtw}
    # n-gram definition
    for i, _ in enumerate(log_df):
        ac_n_grams = list(ngrams(log_df[i]['ac_order'], n_size,
                                 pad_left=True, left_pad_symbol=0))
        rl_n_grams = list(ngrams(log_df[i]['rl_order'], n_size,
                                 pad_left=True, left_pad_symbol=0))
        tn_grams = list(ngrams(log_df[i]['tbtw'], n_size,
                               pad_left=True, left_pad_symbol=0))
        st_idx = 0
        if i == 0:
            vec['prefixes']['x_ac_inp'] = np.array([ac_n_grams[0]])
            vec['prefixes']['x_rl_inp'] = np.array([rl_n_grams[0]])
            vec['prefixes']['xt_inp'] = np.array([tn_grams[0]])
            vec['next_evt']['y_ac_inp'] = np.array(ac_n_grams[1][-1])
            vec['next_evt']['y_rl_inp'] = np.array(rl_n_grams[1][-1])
            vec['next_evt']['yt_inp'] = np.array(tn_grams[1][-1])
            st_idx = 1
        for j in range(st_idx, len(ac_n_grams) - 1):
            vec['prefixes']['x_ac_inp'] = np.concatenate((vec['prefixes']['x_ac_inp'],
                                                          np.array([ac_n_grams[j]])), axis=0)
            vec['prefixes']['x_rl_inp'] = np.concatenate((vec['prefixes']['x_rl_inp'],
                                                          np.array([rl_n_grams[j]])), axis=0)
            vec['prefixes']['xt_inp'] = np.concatenate((vec['prefixes']['xt_inp'],
                                                        np.array([tn_grams[j]])), axis=0)
            vec['next_evt']['y_ac_inp'] = np.append(vec['next_evt']['y_ac_inp'],
                                                    np.array(ac_n_grams[j + 1][-1]))
            vec['next_evt']['y_rl_inp'] = np.append(vec['next_evt']['y_rl_inp'],
                                                    np.array(rl_n_grams[j + 1][-1]))
            vec['next_evt']['yt_inp'] = np.append(vec['next_evt']['yt_inp'],
                                                  np.array(tn_grams[j + 1][-1]))

    vec['prefixes']['xt_inp'] = vec['prefixes']['xt_inp'].reshape(
        (vec['prefixes']['xt_inp'].shape[0],
         vec['prefixes']['xt_inp'].shape[1], 1))

    # print(vec['prefixes']['x_ac_inp'])
    vec['next_evt']['y_ac_inp'] = ku.to_categorical(vec['next_evt']['y_ac_inp'],
                                                    num_classes=len(ac_index))

    vec['next_evt']['y_rl_inp'] = ku.to_categorical(vec['next_evt']['y_rl_inp'],
                                                    num_classes=len(rl_index))

    return vec


def reformat_events(log_df, ac_index, rl_index):
    """Creates series of activities, roles and relative times per trace.
    Args:
        log_df: dataframe.
        ac_index (dict): index of activities.
        rl_index (dict): index of roles.
    Returns:
        list: lists of activities, roles and relative times.
    """
    log_df = log_df.to_dict('records')

    temp_data = list()
    log_df = sorted(log_df, key=lambda x: (x['caseid'], x['end_timestamp']))
    for key, group in itertools.groupby(log_df, key=lambda x: x['caseid']):
        trace = list(group)
        ac_order = [x['ac_index'] for x in trace]
        rl_order = [x['rl_index'] for x in trace]
        tbtw = [x['tbtw_norm'] for x in trace]
        ac_order.insert(0, ac_index[('start')])
        ac_order.append(ac_index[('end')])
        rl_order.insert(0, rl_index[('start')])
        rl_order.append(rl_index[('end')])
        tbtw.insert(0, 0)
        tbtw.append(0)
        temp_dict = dict(caseid=key,
                         ac_order=ac_order,
                         rl_order=rl_order,
                         tbtw=tbtw)
        temp_data.append(temp_dict)

    return temp_data


# =============================================================================
# Support
# =============================================================================


def create_index(log_df, column):
    """Creates an idx for a categorical attribute.
    Args:
        log_df: dataframe.
        column: column name.
    Returns:
        index of a categorical attribute pairs.
    """
    temp_list = log_df[[column]].values.tolist()
    subsec_set = {(x[0]) for x in temp_list}
    subsec_set = sorted(list(subsec_set))
    alias = dict()
    for i, _ in enumerate(subsec_set):
        alias[subsec_set[i]] = i + 1
    return alias


def max_serie(log_df, serie):
    """Returns the max and min value of a column.
    Args:
        log_df: dataframe.
        serie: name of the serie.
    Returns:
        max and min value.
    """
    max_value, min_value = 0, 0
    for record in log_df:
        if np.max(record[serie]) > max_value:
            max_value = np.max(record[serie])
        if np.min(record[serie]) > min_value:
            min_value = np.min(record[serie])
    return max_value, min_value

'''
def max_min_std(val, max_value, min_value):
    """Standardize a number between range.
    Args:
        val: Value to be standardized.
        max_value: Maximum value of the range.
        min_value: Minimum value of the range.
    Returns:
        Standardized value between 0 and 1.
    """
    std = (val - min_value) / (max_value - min_value)
    return std
'''
