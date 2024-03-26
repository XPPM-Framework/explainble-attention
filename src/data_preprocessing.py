from typing import Tuple, Dict, List, Any
from functools import partial

import networkx as nx
import numpy as np
import pandas as pd
import keras.utils as ku

from data_reader import LogReader, find_index, build_profile, det_freq_matrix, \
    det_correlation_matrix, connected_component_subgraphs, role_definition, graph_network
from nn_support import reduce_loops, add_calculated_features, create_index, vectorization


def preprocess_dataframe(log_df: pd.DataFrame, *,
                         no_loops: bool, norm_method: str, n_size: int,
                         **kwargs) -> Tuple[dict, dict, dict, np.ndarray, np.ndarray]:
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


def create_resource_roles(log: LogReader, *, perform_role_mining: bool = True) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Discovers and adds resource roles (groups) to each row in the event log. Also gives a mapping of from resource to role.
    :param log:
    :param perform_role_mining: Whether to actually perform role mining or just provide a resource mapping.
    :return:
    """
    _, resource_table = read_resource_pool(log, sim_percentage=0.50)
    # Role discovery
    log_df_resources = pd.DataFrame.from_records(resource_table)
    log_df_resources = log_df_resources.rename(index=str, columns={"resource": "user"})
    role_mapping: Dict[str, str] = {resource: role for role, resource in map(lambda x: x.values(), resource_table)}

    # Hacky way to not propagate the done role mining while affecting the rest of the workflow as little as possible
    if not perform_role_mining:
        # Override with identity mapping for each resource
        log_df_resources["role"] = log_df_resources.loc[:, "user"]
        role_mapping: Dict[str, str] = {resource: resource for role, resource in map(lambda x: x.values(), resource_table)}
        print("**Not** performing role mining")

    # Dataframe creation
    log_df = pd.DataFrame.from_records(log.data)
    log_df = log_df.merge(log_df_resources, on='user', how='left')
    log_df = log_df[log_df.task != 'Start']
    log_df = log_df[log_df.task != 'End']
    log_df = log_df.reset_index(drop=True)

    # Create pickleable mapping from resource to role

    return log_df, role_mapping


def add_resource_roles(log_df: pd.DataFrame, role_mapping: Dict[str, str], *,
                       resource_column: str = "user") -> pd.DataFrame:
    """
    Add the 'role' column based on the resource ('user' column) and the given role mapping.
    :param log_df: The dataframe to add the 'role' columns to.
    :param role_mapping: Dictionary mapping resource -> role. Needs to map all resources.
    :param resource_column: The name of the column containing the resources. Defaults to 'user'.
    :return:
    """
    log_df["role"] = log_df[resource_column].map(lambda resource: role_mapping[resource])
    return log_df


def read_resource_pool(log, separator=None, drawing=False, sim_percentage=0.7):
    if separator == None:
        filtered_list = list()
        for row in log.data:
            if row['task'] != 'End' and row['user'] != 'AUTO':
                filtered_list.append([row['task'], row['user']])
        return role_discovery(filtered_list, drawing, sim_percentage)
    else:
        raw_list = list()
        filtered_list = list()
        for row in log.data:
            raw_list.append(row['user'])
        filtered_list = list(set(raw_list))
        return read_roles_from_columns(raw_list, filtered_list, separator)


def role_discovery(data, drawing, sim_percentage):
    tasks = list(set(list(map(lambda x: x[0], data))))
    try:
        tasks.remove('Start')
    except Exception:
        pass
    tasks = [dict(index=i, data=tasks[i]) for i in range(0, len(tasks))]
    users = list(set(list(map(lambda x: x[1], data))))
    try:
        users.remove('Start')
    except Exception:
        pass
    users = [dict(index=i, data=users[i]) for i in range(0, len(users))]
    data_transform = list(map(lambda x: [find_index(tasks, x[0]), find_index(users, x[1])], data))
    unique = list(set(tuple(i) for i in data_transform))
    unique = [list(i) for i in unique]
    # [print(uni) for uni in users]
    # building of a task-size profile of task execution per resource
    profiles = build_profile(users, det_freq_matrix(unique, data_transform), len(tasks))
    print('Analysing resource pool...')
    #    sup.print_progress(((20 / 100)* 100),'Analysing resource pool ')
    # building of a correlation matrix between resouces profiles
    correlation_matrix = det_correlation_matrix(profiles)
    #    sup.print_progress(((40 / 100)* 100),'Analysing resource pool ')
    # creation of a relation network between resouces
    g = nx.Graph()
    for user in users:
        g.add_node(user['index'])
    for relation in correlation_matrix:
        # creation of edges between nodes excluding the same element correlation
        # and those below the 0.7 threshold of similarity
        if relation['distance'] > sim_percentage and relation['x'] != relation['y']:
            g.add_edge(relation['x'], relation['y'], weight=relation['distance'])
    #    sup.print_progress(((60 / 100)* 100),'Analysing resource pool ')
    # extraction of fully conected subgraphs as roles
    sub_graphs = list(connected_component_subgraphs(g))
    #    sup.print_progress(((80 / 100)* 100),'Analysing resource pool ')
    # role definition from graph
    roles = role_definition(sub_graphs, users)
    # plot creation (optional)
    if drawing == True:
        graph_network(g, sub_graphs)
    #    sup.print_progress(((100 / 100)* 100),'Analysing resource pool ')
    return roles


def read_roles_from_columns(raw_data, filtered_data, separator):
    records = list()
    role_list = list()
    pool_list = list()
    raw_splited = list()
    for row in raw_data:
        temp = row.split(separator)
        if temp[0] != 'End':
            raw_splited.append(dict(role=temp[1], resource=temp[0]))
    for row in filtered_data:
        temp = row.split(separator)
        if temp[0] != 'End':
            pool_list.append(dict(role=temp[1], resource=temp[0]))
            role_list.append(temp[1])
    role_list = list(set(role_list))
    for role in role_list:
        members = list(filter(lambda person: person['role'] == role, pool_list))
        members = list(map(lambda x: x['resource'], members))
        quantity = len(members)
        # freq = len(list(filter(lambda person: person['role'] == role, raw_splited)))
        records.append(dict(role=role, quantity=quantity, members=members))
    return records


def revert_activity_index_mappings(df: pd.DataFrame, activity_columns: List[str], index_mapping: dict) -> pd.DataFrame:
    """
    Reverts all index mappings of the dataframe.
    :param df: The complete dataframe.
    :param activity_columns: The list of columns containing activity indices go revert.
    :param index_mapping: The mapping of index to activity name.
    :return:
    """

    def map_value_or_list(x: Any, mapping: dict):
        if isinstance(x, list):
            return [mapping[str(v)] for v in x]
        else:
            return index_mapping[str(x)]

    for column in activity_columns:
        df[column] = df[column].map(partial(map_value_or_list, mapping=index_mapping))

    return df
