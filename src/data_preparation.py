from typing import Tuple, Dict

import numpy as np
import pandas as pd
import keras.utils as ku

from data_reader import LogReader, read_resource_pool
from nn_support import reduce_loops, add_calculated_features, create_index, vectorization


def preprocess_dataframe(log_df: pd.DataFrame, *, no_loops: bool, norm_method: str, n_size: int, **kwargs)\
        -> Tuple[dict, dict, dict, np.ndarray, np.ndarray]:
    """
    Prepares the dataset by adding additional features and vectorizing it
    :param log_df: The dataframe to preprocess.
    :param no_loops: Whether to remove loops.
    :param norm_method: What normalization method to use. 'max' or 'lognorm'.
    :param n_size: The n-gram size to use.
    :return: A dictionary of the vectorized dataframe along with the created indices and the embedded matrices for activities and roles.
    \nContains keys 'prefixes', 'next_evt', 'mean_tbtw', 'std_tbtw'
    """

    df = log_df.copy()

    if no_loops:
        df = reduce_loops(df)
    # Index creation
    ac_index = create_index(df, 'task')
    ac_index['start'] = 0
    ac_index['end'] = len(ac_index)
    index_ac = {v: k for k, v in ac_index.items()}

    rl_index = create_index(df, 'role')
    rl_index['start'] = 0
    rl_index['end'] = len(rl_index)
    index_rl = {v: k for k, v in rl_index.items()}

    # Load embedded matrix
    ac_weights = ku.to_categorical(sorted(index_ac.keys()), len(ac_index))
    print('AC_WEIGHTS', ac_weights)
    rl_weights = ku.to_categorical(sorted(index_rl.keys()), len(rl_index))
    print('RL_WEIGHTS', rl_weights)

    # Calculate relative times
    df = add_calculated_features(df, ac_index, rl_index)

    # Input vectorization
    vec = vectorization(df, ac_index, rl_index, norm_method=norm_method, n_size=n_size)

    return vec, index_ac, index_rl, ac_weights, rl_weights


def create_resource_roles(log: LogReader) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Discovers and adds resource roles (groups) to each row in the event log. Also gives a mapping of from resource to role.
    :param log:
    :return:
    """
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

    # Create pickleable mapping from resource to role
    role_mapping: Dict[str, str] = {resource: role for role, resource in map(lambda x: x.values(), resource_table)}

    return log_df, role_mapping


def add_resource_roles(log_df: pd.DataFrame, role_mapping: Dict[str, str], *, resource_column: str = "user") -> pd.DataFrame:
    """
    Add the 'role' column based on the resource ('user' column) and the given role mapping.
    :param log_df: The dataframe to add the 'role' columns to.
    :param role_mapping: Dictionary mapping resource -> role. Needs to map all resources.
    :param resource_column: The name of the column containing the resources. Defaults to 'user'.
    :return:
    """
    log_df["role"] = log_df[resource_column].map(lambda resource: role_mapping[resource])
    return log_df
