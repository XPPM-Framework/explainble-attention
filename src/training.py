import os
from pathlib import Path

import keras.layers as L
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

from data_reader import LogReader
from nn_support import reduce_loops, split_train_test, add_calculated_features, create_index
from util import create_json, read_json, create_csv_file_header, get_parameter_path
from data_preprocessing import preprocess_dataframe

# tf.disable_v2_behavior()

MY_WORKSPACE_DIR = os.getenv("MY_WORKSPACE_DIR", "BPIC_Data/")


def training_model(log_df: pd.DataFrame, args: dict, *, no_loops: bool = False):
    """Main method of the training module.
    :param log_df: The event log as a dataframe.
    :param args: Parameters for training the network.
    :param no_loops: Whether to remove loops fom the event-log (optional).
    """
    vec, index_ac, index_rl, ac_weights, rl_weights = preprocess_dataframe(log_df, no_loops=no_loops, **args)

    # Parameters export
    output_folder = os.path.join(args['folder'])
    print('Passing output_folder======', output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        os.makedirs(os.path.join(output_folder, 'parameters'))

    # Create test set csv file
    # Split validation datasets
    """
    log_df_train, log_df_test = split_train_test(log_df, 0.3)  # 70%/30%
    create_csv_file_header(log_df_test.to_dict('records'),
                           os.path.join(output_folder,
                                        'parameters',
                                        args['log_name'] + 'test_log.csv'))
    """

    # Create json file with the used training prameters
    parameters_path = get_parameter_path(args['model_path'])
    parameters = read_json(parameters_path)
    parameters['event_log'] = args['file_name']
    parameters['exp_desc'] = args
    parameters['index_ac'] = index_ac
    parameters['ac_index'] = {v: k for k, v in index_ac.items()}
    parameters['index_rl'] = index_rl
    parameters['rl_index'] = {v: k for k, v in index_rl.items()}
    parameters['dim'] = dict(samples=str(vec['prefixes']['x_ac_inp'].shape[0]),
                             time_dim=str(vec['prefixes']['x_ac_inp'].shape[1]),
                             features=str(len(index_ac.items())))
    parameters['mean_tbtw'] = vec['mean_tbtw']
    parameters['std_tbtw'] = vec['std_tbtw']
    parameters['incl_time'] = args.get('incl_time', True)
    parameters['incl_res'] = args.get('incl_res', True)
    parameters['perform_role_mining'] = args.get('perform_role_mining', True)

    create_json(parameters, get_parameter_path(model_path=args['model_path']))

    # pickle.dump(vec, open( os.path.join(output_folder,
    #                                        'parameters',
    #                                    args['log_name']+'train_vec.pkl'), "wb"))

    if args['task'] == 'prefix_attn':
        model = training_model_temporal(vec, ac_weights, rl_weights, output_folder, args)
    elif(args['task']=='full_attn'):
        model = training_model_temporal_variable(vec, ac_weights, rl_weights, output_folder, args)
    else:
        raise Exception(f"{args['task']} is not a valid task type")
    """
    Other task types which were commented because we do not use them
    
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
    incl_time = args.get("incl_time", True)
    if incl_time:
        print("Using time component")
    incl_res = args.get("incl_res", True)
    if incl_res:
        print("Using resource component")
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


def training_model_temporal_variable(vec, ac_weights, rl_weights, output_folder, args):

    dropout_input = 0.01
    dropout_context=0.30
    lstm_size_alpha=args['l_size']
    lstm_size_beta=args['l_size']
    print("Training prefix and variable attention model")

    l2reg=0.0001
    allow_negative=False
    incl_time = args.get("incl_time", True)
    if incl_time:
        print("Using time component")
    incl_res = args.get("incl_res", True)
    if incl_res:
        print("Using resource component")
    #Code Input
    ac_input = Input(shape=(vec['prefixes']['x_ac_inp'].shape[1], ), name='ac_input')
    rl_input = Input(shape=(vec['prefixes']['x_rl_inp'].shape[1], ), name='rl_input')
    t_input = Input(shape=(vec['prefixes']['xt_inp'].shape[1], 1), name='t_input')



    ########################################



    #inputs_list = [ac_input]

    #Calculate embedding for each code and sum them to a visit level
    ac_embedding = L.Embedding(ac_weights.shape[0],
                               ac_weights.shape[1],
                               weights=[ac_weights],
                               input_length=vec['prefixes']['x_ac_inp'].shape[1],
                               trainable=True, name='ac_embedding')(ac_input)

    dim =ac_weights.shape[1]

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

        #Apply dropout on inputs
    full_embs = L.Dropout(dropout_input)(full_embs)

    if incl_time==True:
        time_embs = L.concatenate([full_embs, t_input], name='allInp')

        dim += 1
    else:
        time_embs=full_embs

    #Numeric input if needed
    alpha = L.Bidirectional(L.LSTM(lstm_size_alpha, return_sequences=True),
                            name='alpha')
    beta = L.Bidirectional(L.LSTM(lstm_size_beta, return_sequences=True),
                           name='beta')
    alpha_dense = L.Dense(1, kernel_regularizer=l2(l2reg))
    beta_dense = L.Dense(dim,
                         activation='tanh', kernel_regularizer=l2(l2reg))

    #Compute alpha, visit attention
    alpha_out = alpha(time_embs)
    alpha_out = L.TimeDistributed(alpha_dense, name='alpha_dense_0')(alpha_out)
    alpha_out = L.Softmax(axis=1, name='alpha_softmax')(alpha_out)
    #Compute beta, codes attention
    beta_out = beta(time_embs)
    beta_out = L.TimeDistributed(beta_dense, name='beta_dense_0')(beta_out)
    #Compute context vector based on attentions and embeddings
    c_t = L.Multiply()([alpha_out, beta_out, time_embs])
    c_t = L.Lambda(lambda x: K.sum(x, axis=1))(c_t)
    #Reshape to 3d vector for consistency between Many to Many and Many to One implementations
    #contexts = L.Lambda(reshape)(c_t)

    #Make a prediction
    contexts = L.Dropout(dropout_context)(c_t)

    act_output = Dense(ac_weights.shape[0],
                       activation='softmax',
                       kernel_initializer='glorot_uniform',
                       name='act_output')(contexts)


    model = Model(inputs=[ac_input, rl_input, t_input], outputs=act_output)

    if args['optim'] == 'Nadam':
        opt = Nadam(lr=0.0005, beta_1=0.9, beta_2=0.999,
                    epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
    elif args['optim'] == 'Adam':
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
                   epsilon=None, decay=0.0, amsgrad=False)
    elif args['optim'] == 'SGD':
        opt = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    elif args['optim'] == 'Adagrad':
        opt = Adagrad(lr=0.01, epsilon=None, decay=0.0)

    model.compile(loss={'act_output':'categorical_crossentropy'}, optimizer=opt)

    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=42)
    #
    #    # Output file
    output_file_path = os.path.join(output_folder,
                                    'models/model_rd_' + str(args['n_size']) +
                                    ' ' + args['optim']  + args['log_name']  +
                                    '_{epoch:02d}-{val_loss:.2f}.h5')
    print('This is the output file path ', output_file_path)
    # Saving
    model_checkpoint = ModelCheckpoint(output_file_path,
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

    #model.fit({'ac_input':, 'rl_input':, 't_input':},
    model.fit(model_inputs,
              {'act_output':vec['next_evt']['y_ac_inp']},
              validation_split=0.2,
              verbose=2,
              callbacks=[early_stopping, model_checkpoint, lr_reducer],
              batch_size=50,
              epochs=100)
    return model
