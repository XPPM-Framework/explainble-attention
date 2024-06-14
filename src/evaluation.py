"""
Taken from the last notebook cell, allegedly specifically for next activity prediction
"""
from typing import Tuple

# -*- coding: utf-8 -*-
##### Reused existing code for post processing with modifications to get attention weights
"""
Created on Fri Mar  8 08:16:15 2019

@author: Manuel Camargo
"""
import json
import os
import math

from keras.models import load_model, Model

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from util import create_csv_file, create_csv_file_header, plot_history, get_parameter_path
from nn_support import reduce_loops, add_calculated_features, create_index, vectorization

MY_WORKSPACE_DIR = os.getenv("MY_WORKSPACE_DIR", "../")
START_TIMEFORMAT = ''
INDEX_AC = None
INDEX_RL = None
DIM = dict()
TBTW = dict()
EXP = dict()
timeformat = '%Y-%m-%dT%H:%M:%S.%f'


def predict_next(dataframe: pd.DataFrame, timeformat: str, parameters: dict,
                 *, start_timestamp_col: str = "time:timestamp", end_timestamp_col: str = "time:timestamp",
                 no_loops: bool = False,
                 is_single_exec=True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Main function of the suffix prediction module.
    Args:
        timeformat (str): event-log date-time format.
        parameters (dict): parameters used in the training step.
        is_single_exec (boolean): generate measurments stand alone or share
                    results with other runing experiments (optional)
    :param start_timestamp_col:
    :param end_timestamp_col:
    """
    global START_TIMEFORMAT
    global INDEX_AC
    global INDEX_RL
    global DIM
    global TBTW
    global EXP

    START_TIMEFORMAT = timeformat

    output_route = parameters['folder']
    model_name, _ = os.path.splitext(parameters['model_path'])
    parameters_path = get_parameter_path(parameters['model_path'])

    # Loading of parameters from training
    with open(parameters_path) as file:
        data = json.load(file)
        EXP = {k: v for k, v in data['exp_desc'].items()}
        print(EXP)
        DIM['samples'] = int(data['dim']['samples'])
        DIM['time_dim'] = int(data['dim']['time_dim'])
        DIM['features'] = int(data['dim']['features'])

        TBTW['mean_tbtw'] = float(data['mean_tbtw'])
        INDEX_AC = {int(k): v for k, v in data['index_ac'].items()}
        INDEX_RL = {int(k): v for k, v in data['index_rl'].items()}
        file.close()

    ###############Load the training data for LIME/SHAP
    # train_vec = pickle.load(open( os.path.join(output_route,
    #                                        'parameters',
    #                                      parameters['log_name']+ 'train_vec.pkl'), 'rb'))

    # ac_input = train_vec['prefixes']['x_ac_inp']

    # rl_input = train_vec['prefixes']['x_rl_inp']
    # t_input = train_vec['prefixes']['xt_inp']
    # y_train = np.argmax(train_vec['next_evt']['y_ac_inp'], axis=1)
    # x_train=ac_input

    # Parameters
    log_parameters = parameters["log_parameters"]
    case_id_key = log_parameters["case_id_key"]
    activity_key = log_parameters["activity_key"]
    timestamp_key = log_parameters["timestamp_key"]
    #resource_key = log_parameters["resource_key"]

    # Loading of testing dataframe
    df_test = dataframe
    df_test['start_timestamp'] = pd.to_datetime(df_test['start_timestamp'])
    df_test['end_timestamp'] = pd.to_datetime(df_test['end_timestamp'])

    # Preprocess
    if no_loops:
        df_test = reduce_loops(df_test)
    ac_index = parameters['ac_index']
    rl_index = parameters['rl_index']
    df_test = add_calculated_features(df_test, ac_index, rl_index)
    df_test = df_test.drop(columns=['user'])
    df_test = df_test.rename(index=str, columns={"role": "user"})

    if EXP['norm_method'] == 'max':
        mean_tbtw = np.mean(df_test.tbtw)
        std_tbtw = np.std(df_test.tbtw)
        norm = lambda x: (x['tbtw'] - mean_tbtw) / std_tbtw
        df_test['tbtw_norm'] = df_test.apply(norm, axis=1)
    elif EXP['norm_method'] == 'lognorm':
        logit = lambda x: math.log1p(x['tbtw'])
        df_test['tbtw_log'] = df_test.apply(logit, axis=1)
        mean_tbtw = np.mean(df_test.tbtw_log)
        std_tbtw = np.std(df_test.tbtw_log)
        norm = lambda x: (x['tbtw_log'] - mean_tbtw) / std_tbtw
        df_test['tbtw_norm'] = df_test.apply(norm, axis=1)

    #   Next event selection method and numbers of repetitions
    variants = [{'imp': 'Arg Max', 'rep': 1}]  # ,
    # {'imp': 'Random Choice', 'rep': 1}]
    #   Generation of predictions
    has_time = False
    model = load_model(os.path.join(output_route, parameters['model_path']))
    layer_names = [layer.name for layer in model.layers]
    print(layer_names)
    rl_emb_weights = None
    ac_emb_weights = model.get_layer(name='ac_embedding').get_weights()[0]
    if 'rl_embedding' in layer_names:
        rl_emb_weights = model.get_layer(name='rl_embedding').get_weights()[0]
    if 't_input' in layer_names:
        has_time = True
    # print(rl_emb_weights)
    ac_output_weights, ac_bias = model.get_layer(name='act_output').get_weights()
    print(ac_output_weights)
    prefix_only = False
    if (parameters['attention'] == 'prefix'):
        model_with_attention = Model(model.inputs, model.outputs + \
                                     [model.get_layer(name='alpha_softmax').output])
        prefix_only = True
    else:
        model_with_attention = Model(model.inputs, model.outputs + \
                                     [model.get_layer(name='alpha_softmax').output, \
                                      model.get_layer(name='beta_dense_0').output])
    temporal_vectors = []
    for var in variants:
        measurements = list()
        print(var['imp'])
        prefixes = create_pref_suf(df_test, ac_index, rl_index)
        # if temporal attention True, else False

        prefixes, temporal_vectors, variable_vectors = predict_next_in(model_with_attention, ac_emb_weights,
                                                                       rl_emb_weights, ac_output_weights, has_time,
                                                                       prefixes, var['imp'], prefix_only)

        accuracy = (np.sum([x['ac_true'] for x in prefixes]) / len(prefixes))
        print("accuracy:" + str(accuracy))
        y_pred = [x['ac_pred'] for x in prefixes]
        y_true = [x['ac_next'] for x in prefixes]

        from sklearn.metrics import classification_report
        print(classification_report(y_true, y_pred))

        file_name = 'results/' + parameters['log_name'] + 'next_event_measures.csv'
        # Save results
        measurements.append({**dict(model=os.path.join(output_route, file_name),
                                    implementation=var['imp']), **{'accuracy': accuracy},
                             **EXP})
        if measurements:
            if os.path.exists(os.path.join(output_route, file_name)):
                create_csv_file(measurements, os.path.join(output_route, file_name), mode='a')
            else:
                create_csv_file_header(measurements, os.path.join(output_route, file_name))

    # print(attention_vector_final)
    file_name = parameters['log_name'] + str(DIM['time_dim'])
    path = output_route + '/results/'
    temp_final = np.mean(np.array(temporal_vectors), axis=0)
    pd.DataFrame(temp_final, columns=['alpha attention weight']).plot(kind='bar',
                                                                      title='Attention of '
                                                                            ' index')
    plot_history(plt, file_name + 'prefix_attn', path)
    plt.show()

    temporal_attention_df = pd.DataFrame(temporal_vectors)

    df_global_attribute_attention = df_local_attribute_attention = None
    if (len(variable_vectors) > 0):
        var_final = np.mean(np.array(variable_vectors), axis=0)

        ac_labels = [INDEX_AC[key] for key in sorted(INDEX_AC.keys())]
        rl_labels = [INDEX_RL[key] for key in sorted(INDEX_RL.keys())]

        num_dim = var_final.shape[0]
        print(num_dim)

        if rl_emb_weights is not None:
            ac_labels.extend(rl_labels)
        if (num_dim == len(ac_labels) + 1):
            ac_labels.append('time')

        df_var = pd.DataFrame({'attribute_values': ac_labels, 'attributes': var_final})
        df_var.plot.bar(y='attributes', x='attribute_values',
                        title='Attention of the event attributes.', figsize=(10, 5))

        plot_history(plt, file_name + 'variable_attn', path)

        plt.show()

        # map for proper naming of returned dataframe
        df_global_attribute_attention = df_var.rename(
            {"attributes": "attention", "attribute_values": "attributes"}).set_index("attributes")

        df_local_attribute_attention = pd.DataFrame(variable_vectors, columns=ac_labels)

    prefix_df = pd.DataFrame.from_records(prefixes).rename({
        "ac": activity_key,
        "ac_pref": "Prefix", "ac_next": "Next Activity - Ground Truth", "ac_pred": "Next Activity - Prediction",
        "ac_true": "Correct Prediction", "rl_pref": "Role Prefix", "rl_next": "Next Role",
        "t_pref": "Time Between Prefix",
    }, axis=1)
    # Get the first occurrence of each case id
    first_event_per_case_indices = df_test.groupby("caseid").head(1).index
    # case_ids = df_test.drop(first_event_per_case_indices, axis=0)["caseid"]
    case_ids = df_test["caseid"]
    # Add case ids to prefix_df
    case_id_key = parameters["log_parameters"]["case_id_key"]
    prefix_df.insert(0, case_id_key, case_ids.values)
    # Reorder columns
    prefix_df = prefix_df.filter([case_id_key, activity_key, "Prefix", "Next Activity - Ground Truth",
                                  "Next Activity - Prediction", "Correct Prediction", "Role Prefix", "Next Role",
                                  "Time Between Prefix"])

    return temporal_attention_df, df_global_attribute_attention, df_local_attribute_attention, prefix_df,

        # var_local = np.array(variable_vectors[2919])
        # df_var_local=pd.DataFrame({'attributes':var_local, 'attribute_values':ac_labels})
        # df_var_local.plot.bar(y='attributes', x='attribute_values',
        #                          title='Attention of the event attributes.')

        # plt.show()
        # plotting local explanation
        # local_var = np.array(temporal_vectors[2919])
        # pd.DataFrame(local_var, columns=['alpha attention weight']).plot(kind='bar',
        #                                                                  title='Attention of '
        #                                                                         ' index')
        # plt.show()


# =============================================================================
# Predict traces
# =============================================================================

def predict_next_in(model_attn, ac_emb_weights, rl_emb_weights, ac_output_weights, has_time, prefixes, imp,
                    prefix_only=False):
    """Generate business process suffixes using a keras trained model.
    Args:
        model (keras model): keras trained model.
        prefixes (list): list of prefixes.
        ac_index (dict): index of activities.
        rl_index (dict): index of roles.
        imp (str): method of next event selection.
    """
    # Generation of predictions
    temporal_vectors = []
    variable_vectors = []
    x_test = []
    y_test = []
    t_dim = DIM['time_dim']
    f_dim = DIM['features']
    x_test_pos = np.empty((0, t_dim, f_dim))
    x_test_neg = np.empty((0, t_dim, f_dim))

    for prefix in prefixes:

        # Activities and roles input shape(1,5)
        x_ac_ngram = np.append(
            np.zeros(DIM['time_dim']),
            np.array(prefix['ac_pref']),
            axis=0)[-DIM['time_dim']:].reshape((1, DIM['time_dim']))

        x_rl_ngram = np.append(
            np.zeros(DIM['time_dim']),
            np.array(prefix['rl_pref']),
            axis=0)[-DIM['time_dim']:].reshape((1, DIM['time_dim']))

        # times input shape(1,5,1)
        x_t_ngram = np.array([np.append(
            np.zeros(DIM['time_dim']),
            np.array(prefix['t_pref']),
            axis=0)[-DIM['time_dim']:].reshape((DIM['time_dim'], 1))])

        betas = None
        # proba = model.predict(x_ac_ngram)

        if prefix_only:
            proba, alphas = model_attn.predict([x_ac_ngram, x_rl_ngram, x_t_ngram])
        else:
            if rl_emb_weights is not None and has_time == True:
                proba, alphas, betas = model_attn.predict([x_ac_ngram, x_rl_ngram, x_t_ngram])
            elif rl_emb_weights is not None and has_time == False:
                proba, alphas, betas = model_attn.predict([x_ac_ngram, x_rl_ngram])
            else:
                proba, alphas, betas = model_attn.predict([x_ac_ngram])
        # print(proba, alphas, betas)
        proba = np.squeeze(proba)
        alphas = np.squeeze(alphas)
        temporal_att_vec = alphas
        assert (np.sum(temporal_att_vec) - 1.0) < 1e-5
        # print(temporal_att_vec)
        temporal_vectors.append(temporal_att_vec)

        if betas is not None:
            # get the beta value
            betas = np.squeeze(betas)
            idx = np.argmax(alphas)
            # print(idx)
            beta_val = betas[idx]
            # get the activity and role for that idx
            act_ip = int(x_ac_ngram[0][idx])
            ac_emb = ac_emb_weights[act_ip]
            dim = ac_emb.shape[0]
            emb = ac_emb

            if rl_emb_weights is not None:
                rol_ip = int(x_rl_ngram[0][idx])
                r_emb = rl_emb_weights[rol_ip]
                dim = dim + r_emb.shape[0]
                emb = np.concatenate((ac_emb, r_emb), axis=None)

            if (betas.shape[1] == dim + 1):
                time_v = np.squeeze(x_t_ngram)[idx]  # time and role as masked together
                emb = np.concatenate((ac_emb, r_emb, time_v), axis=None)

            # print('beta_val',beta_val.shape)
            beta_scaled = np.multiply(beta_val, emb)
            variable_attn = alphas[idx] * beta_scaled
            # sum_grad = np.sum(ac_output_weights, axis=1)
            # variable_attn=np.multiply(sum_grad.flatten(), variable_attn)

            variable_vectors.append(variable_attn)

        if imp == 'Random Choice':
            # Use this to get a random choice following as PDF the predictions
            pos = np.argmax(proba)

        elif imp == 'Arg Max':
            # Use this to get the max prediction
            pos = np.argmax(proba)

        prefix['ac_pred'] = pos
        # Activities accuracy evaluation
        if pos == prefix['ac_next']:
            prefix['ac_true'] = 1
            #if (idx < 4):
            #    print('value is ', len(variable_vectors))
        else:
            prefix['ac_true'] = 0
        # x_test_neg = np.append(x_test_neg,x_ac_ngram, axis=0)
        ####get the temporal attention

        # attention_vector_final = np.mean(np.array(attention_vectors), axis=0)
        # plot part.

    # print_done_task()
    return prefixes, temporal_vectors, variable_vectors


# =============================================================================
# Reformat
# =============================================================================
def create_pref_suf(df_test: pd.DataFrame, ac_index, rl_index):
    """Extraction of prefixes and expected suffixes from event log.
    Args:
        df_test (dataframe): testing dataframe in pandas format.
        ac_index (dict): index of activities.
        rl_index (dict): index of roles.
        pref_size (int): size of the prefixes to extract.
    Returns:
        list: list of prefixes and expected sufixes.
    """
    prefixes = list()
    cases = df_test.caseid.unique()
    for case in cases:
        trace = df_test[df_test.caseid == case]
        ac_pref = list()
        rl_pref = list()
        t_pref = list()
        for i in range(0, len(trace)):
            ac = trace.iloc[i]['ac_index']
            ac_pref.append(trace.iloc[i]['ac_index'])
            rl_pref.append(trace.iloc[i]['rl_index'])
            t_pref.append(trace.iloc[i]['tbtw_norm'])
            prefixes.append(dict(ac=ac,
                                 ac_pref=ac_pref.copy(),
                                 ac_next=trace.iloc[i + 1]['ac_index'] if i + 1 < len(trace) else ac_index["end"],
                                 rl_pref=rl_pref.copy(),
                                 rl_next=trace.iloc[i + 1]['rl_index'] if i + 1 < len(trace) else rl_index["end"],
                                 t_pref=t_pref.copy()))
    return prefixes


def ae_measure(prefixes):
    """Absolute Error measurement.
    Args:
        prefixes (list): list with predicted remaining-times and expected ones.
    Returns:
        list: list with measures added.
    """
    for prefix in prefixes:
        prefix['ae'] = abs(prefix['rem_time'] - prefix['rem_time_pred'])
    return prefixes
