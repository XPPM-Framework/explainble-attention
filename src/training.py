import os
from pathlib import Path

import keras.layers as L
import keras.utils as ku
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Nadam, Adam, SGD, Adagrad
from keras.regularizers import l2

from data_reader import LogReader, read_resource_pool
from nn_support import reduce_loops, split_train_test, add_calculated_features, vectorization, create_index
from util import create_json, create_csv_file_header

# tf.disable_v2_behavior()

MY_WORKSPACE_DIR = os.getenv("MY_WORKSPACE_DIR", "BPIC_Data/")


def training_model(log: LogReader, timeformat: str, args: dict, *, no_loops=False):
    """Main method of the training module.
    Args:
        timeformat (str): event-log date-time format.
        args (dict): parameters for training the network.
        no_loops (boolean): remove loops fom the event-log (optional).
    """
    parameters = dict()
    # read the logfile
    _, resource_table = read_resource_pool(log, sim_percentage=0.50)
    # Role discovery
    log_df_resources = pd.DataFrame.from_records(resource_table)
    log_df_resources = log_df_resources.rename(index=str, columns={"resource": "user"})
    # Dataframe creation
    log_df = pd.DataFrame.from_records(log.data)
    log_df = log_df.merge(log_df_resources, on='user', how='left')
    log_df = log_df[log_df.task != 'Start']
    log_df = log_df[log_df.task != 'End']
    log_df = log_df.reset_index(drop=True)

    if no_loops:
        log_df = reduce_loops(log_df)
    # Index creation
    ac_index = create_index(log_df, 'task')
    ac_index['start'] = 0
    ac_index['end'] = len(ac_index)
    index_ac = {v: k for k, v in ac_index.items()}

    rl_index = create_index(log_df, 'role')
    rl_index['start'] = 0
    rl_index['end'] = len(rl_index)
    index_rl = {v: k for k, v in rl_index.items()}

    # Load embedded matrix
    ac_weights = ku.to_categorical(sorted(index_ac.keys()), len(ac_index))
    print('AC_WEIGHTS', ac_weights)
    rl_weights = ku.to_categorical(sorted(index_rl.keys()), len(rl_index))
    print('RL_WEIGHTS', rl_weights)

    # Calculate relative times
    log_df = add_calculated_features(log_df, ac_index, rl_index)
    # Split validation datasets
    log_df_train, log_df_test = split_train_test(log_df, 0.3)  # 70%/30%
    # Input vectorization
    vec = vectorization(log_df_train, ac_index, rl_index, args)
    # print(vec['prefixes']['x_ac_inp'])

    # Parameters export
    output_folder = os.path.join(args['folder'])
    print('Passing output_folder======', output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        os.makedirs(os.path.join(output_folder, 'parameters'))

    parameters['event_log'] = args['file_name']
    parameters['exp_desc'] = args
    parameters['index_ac'] = index_ac
    parameters['index_rl'] = index_rl
    parameters['dim'] = dict(samples=str(vec['prefixes']['x_ac_inp'].shape[0]),
                             time_dim=str(vec['prefixes']['x_ac_inp'].shape[1]),
                             features=str(len(ac_index)))
    parameters['mean_tbtw'] = vec['mean_tbtw']
    parameters['std_tbtw'] = vec['std_tbtw']

    create_json(parameters, os.path.join(output_folder,
                                         'parameters',
                                         args['log_name'] + 'model_parameters.json'))

    # pickle.dump(vec, open( os.path.join(output_folder,
    #                                        'parameters',
    #                                    args['log_name']+'train_vec.pkl'), "wb"))

    create_csv_file_header(log_df_test.to_dict('records'),
                           os.path.join(output_folder,
                                        'parameters',
                                        args['log_name'] + 'test_log.csv'))

    if args['task'] == 'prefix_attn':
        model = training_model_temporal(vec, ac_weights, rl_weights, output_folder, args)
    else:
        raise Exception(f"{args['task']} is not a valid task type")
    """
    Other task types which were commented because we do not use them
    
    elif(args['task']=='full_attn'):
        model = training_model_temporal_variable(vec, ac_weights, rl_weights, output_folder, args)
    else:
        model = training_model_with_time_prediction(vec, ac_weights, rl_weights, output_folder, args)
    """
    # elif args['model_type'] == 'shared_cat':
    #    training_model_sharedcat(vec, ac_weights, rl_weights, output_folder, args)

    return model


def training_model_temporal(vec, ac_weights, rl_weights, output_folder, args):
    MAX_LEN = args['n_size']
    dropout_input = 0.15
    dropout_context = 0.15
    # number of lstm cells
    incl_time = True
    incl_res = True
    lstm_size_alpha = args['l_size']
    print("Training prefix-attention model")

    l2reg = 0.0001

    # Inputs include activity, resource and time - time is normalised- 0 mean and unit variance
    ac_input = Input(shape=(vec['prefixes']['x_ac_inp'].shape[1],), name='ac_input')
    rl_input = Input(shape=(vec['prefixes']['x_rl_inp'].shape[1],), name='rl_input')
    t_input = Input(shape=(vec['prefixes']['xt_inp'].shape[1], 1), name='t_input')

    ac_embedding = L.Embedding(ac_weights.shape[0],
                               ac_weights.shape[1],
                               weights=[ac_weights],
                               input_length=vec['prefixes']['x_ac_inp'].shape[1],
                               trainable=True, name='ac_embedding')(ac_input)

    dim = ac_weights.shape[1]

    if incl_res:
        rl_embedding = Embedding(rl_weights.shape[0],
                                 rl_weights.shape[1],
                                 weights=[rl_weights],
                                 input_length=vec['prefixes']['x_rl_inp'].shape[1],
                                 trainable=True, name='rl_embedding')(rl_input)
        full_embs = L.concatenate([ac_embedding, rl_embedding], name='catInp')
        dim += rl_weights.shape[1]

    else:
        full_embs = ac_embedding

        # Apply dropout on inputs
    full_embs = L.Dropout(dropout_input)(full_embs)

    if incl_time == True:
        time_embs = L.concatenate([full_embs, t_input], name='allInp')

        dim += 1
    else:
        time_embs = full_embs

    alpha = L.Bidirectional(L.LSTM(lstm_size_alpha, return_sequences=True),
                            name='alpha')
    alpha_dense = L.Dense(1, kernel_regularizer=l2(l2reg))

    # Compute alpha, timestep attention
    alpha_out = alpha(time_embs)
    alpha_out = L.TimeDistributed(alpha_dense, name='alpha_dense_0')(alpha_out)
    alpha_out = L.Softmax(name='alpha_softmax', axis=1)(alpha_out)

    # Compute context vector based on attentions and embeddings
    c_t = L.Multiply()([alpha_out, time_embs])
    c_t = L.Lambda(lambda x: K.sum(x, axis=1))(c_t)

    contexts = L.Dropout(dropout_context)(c_t)

    act_output = Dense(ac_weights.shape[1],
                       activation='softmax',
                       kernel_initializer='glorot_uniform',
                       name='act_output')(contexts)

    model = Model(inputs=[ac_input, rl_input, t_input], outputs=act_output)

    if args['optim'] == 'Nadam':
        opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999,
                    epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
    elif args['optim'] == 'Adam':
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
                   epsilon=None, decay=0.0, amsgrad=False)
    elif args['optim'] == 'SGD':
        opt = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    elif args['optim'] == 'Adagrad':
        opt = Adagrad(lr=0.01, epsilon=None, decay=0.0)

    model.compile(loss={'act_output': 'categorical_crossentropy'}, optimizer=opt, metrics=['accuracy'])

    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=42)
    #
    #    # Output file
    model_file_path = Path(output_folder) / ('models/model_rd_' + str(args['n_size']) +
                                              ' ' + args['optim'] + args['log_name'] +
                                              '_{epoch:02d}-{val_loss:.2f}.h5')
    model_file_path.parent.mkdir(parents=True, exist_ok=True)
    model_file_path = str(model_file_path)
    print(f'Model file saved to "{model_file_path}"')

    # Saving
    model_checkpoint = ModelCheckpoint(model_file_path,
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=False,
                                       mode='auto')
    lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.5,
                                   patience=10,
                                   verbose=0,
                                   mode='auto',
                                   min_delta=0.0001,
                                   cooldown=0,
                                   min_lr=0)
    model_inputs = [vec['prefixes']['x_ac_inp']]
    model_inputs.append(vec['prefixes']['x_rl_inp'])
    model_inputs.append(vec['prefixes']['xt_inp'])

    # model.fit({'ac_input':, 'rl_input':, 't_input':},
    model.fit(model_inputs,
              {'act_output': vec['next_evt']['y_ac_inp']},
              validation_split=0.15,
              verbose=2,
              callbacks=[early_stopping, model_checkpoint, lr_reducer],
              batch_size=100,
              epochs=100)
    return model