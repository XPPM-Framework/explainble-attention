# -*- coding: utf-8 -*-
import csv
import datetime
import xml.etree.ElementTree as ET
import gzip
import zipfile as zf
import os
from pathlib import Path
from typing import Union

import pandas as pd
import pm4py
from tqdm import tqdm

from util import file_size


class LogReader(object):
    """
    This class reads and parse the elements of a given process log in format .xes or .csv
    """

    def __init__(self, input, start_timeformat, end_timeformat, log_columns_numbers=[], ns_include=True,
                 one_timestamp=False):
        """constructor"""
        if not log_columns_numbers:
            log_columns_numbers = []
        self.input = input
        self.data, self.raw_data = self.load_data_from_file(log_columns_numbers, start_timeformat, end_timeformat,
                                                            ns_include, one_timestamp)

    def to_dataframe(self):
        return pd.DataFrame.from_records(self.data)

    # Support Method
    def define_ftype(self):
        filename, file_extension = os.path.splitext(self.input)
        if file_extension == '.xes' or file_extension == '.csv' or file_extension == '.mxml':
            filename = filename + file_extension
            file_extension = file_extension
        elif file_extension == '.gz':
            outFileName = filename
            filename, file_extension = self.decompress_file_gzip(self.input, outFileName)
        elif file_extension == '.zip':
            filename, file_extension = self.decompress_file_zip(self.input, filename)
        elif not (file_extension == '.xes' or file_extension == '.csv' or file_extension == '.mxml'):
            raise IOError('file type not supported')
        return filename, file_extension

    # Decompress .gz files
    def decompress_file_gzip(self, filename, outFileName):
        inFile = gzip.open(filename, 'rb')
        outFile = open(outFileName, 'wb')
        outFile.write(inFile.read())
        inFile.close()
        outFile.close()
        _, fileExtension = os.path.splitext(outFileName)
        return outFileName, fileExtension

    # Decompress .zip files
    def decompress_file_zip(self, filename, outfilename):
        with zf.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall("../inputs/")
        _, fileExtension = os.path.splitext(outfilename)
        return outfilename, fileExtension

    # Reading methods
    def load_data_from_file(self, log_columns_numbers, start_timeformat, end_timeformat, ns_include, one_timestamp):
        """reads all the data from the log depending the extension of the file"""
        temp_data = list()
        filename, file_extension = self.define_ftype()
        if file_extension == '.xes':
            temp_data, raw_data = self.get_xes_events_data(filename, start_timeformat, end_timeformat, ns_include,
                                                           one_timestamp)
        elif file_extension == '.csv':
            temp_data, raw_data = self.get_csv_events_data(log_columns_numbers, start_timeformat, end_timeformat)
        elif file_extension == '.mxml':
            temp_data, raw_data = self.get_mxml_events_data(filename, start_timeformat, end_timeformat)
        return temp_data, raw_data

    def get_xes_events_data(self, filename, start_timeformat, end_timeformat, ns_include, one_timestamp):
        """reads and parse all the events information from a xes file"""
        temp_data = list()
        tree = ET.parse(filename)
        root = tree.getroot()
        if ns_include:
            # TODO revisar como poder cargar el mane space de forma automatica del root
            ns = {'xes': root.tag.split('}')[0].strip('{')}
            tags = dict(trace='xes:trace', string='xes:string', event='xes:event', date='xes:date')
        else:
            ns = {'xes': ''}
            tags = dict(trace='trace', string='string', event='event', date='date')
        traces = root.findall(tags['trace'], ns)
        i = 0
        print('Reading log traces')
        for trace in traces:
            #            sup.print_progress(((i / (len(traces) - 1)) * 100), 'Reading log traces ')
            caseid = ''
            for string in trace.findall(tags['string'], ns):
                if string.attrib['key'] == 'concept:name':
                    caseid = string.attrib['value']
            for event in trace.findall(tags['event'], ns):
                task = ''
                user = ''
                event_type = ''
                complete_timestamp = ''
                for string in event.findall(tags['string'], ns):
                    if string.attrib['key'] == 'concept:name':
                        task = string.attrib['value']
                    if string.attrib['key'] == 'org:resource':
                        user = string.attrib['value']
                    if string.attrib['key'] == 'lifecycle:transition':
                        event_type = string.attrib['value'].lower()
                    if string.attrib['key'] == 'Complete_Timestamp':
                        complete_timestamp = string.attrib['value']
                        if complete_timestamp != 'End':
                            complete_timestamp = datetime.datetime.strptime(complete_timestamp, end_timeformat)
                timestamp = ''
                for date in event.findall(tags['date'], ns):
                    if date.attrib['key'] == 'time:timestamp':
                        timestamp = date.attrib['value']
                        try:
                            timestamp = datetime.datetime.fromisoformat(timestamp[:-6])
                        except ValueError:
                            timestamp = datetime.datetime.fromisoformat(timestamp)
                if not (task == '0' or task == '-1'):
                    temp_data.append(
                        dict(caseid=caseid, task=task, event_type=event_type, user=user, start_timestamp=timestamp,
                             end_timestamp=complete_timestamp))
            i += 1
        raw_data = temp_data
        temp_data = self.reorder_xes(temp_data, one_timestamp)
        return temp_data, raw_data

    def reorder_xes(self, temp_data, one_timestamp):
        """this method joints the duplicated events on the .xes log"""
        ordered_event_log = list()
        if one_timestamp:
            ordered_event_log = list(filter(lambda x: x['event_type'] == 'complete', temp_data))
            for event in ordered_event_log:
                event['end_timestamp'] = event['start_timestamp']
        else:
            events = list(filter(lambda x: (x['event_type'] == 'start' or x['event_type'] == 'complete'), temp_data))
            cases = list({x['caseid'] for x in events})
            for case in cases:
                start_events = sorted(
                    list(filter(lambda x: x['event_type'] == 'start' and x['caseid'] == case, events)),
                    key=lambda x: x['start_timestamp'])
                finish_events = sorted(
                    list(filter(lambda x: x['event_type'] == 'complete' and x['caseid'] == case, events)),
                    key=lambda x: x['start_timestamp'])
                if len(start_events) == len(finish_events):
                    temp_trace = list()
                    for i, _ in enumerate(start_events):
                        match = False
                        for j, _ in enumerate(finish_events):
                            if start_events[i]['task'] == finish_events[j]['task']:
                                temp_trace.append(
                                    dict(caseid=case, task=start_events[i]['task'], event_type=start_events[i]['task'],
                                         user=start_events[i]['user'],
                                         start_timestamp=start_events[i]['start_timestamp'],
                                         end_timestamp=finish_events[j]['start_timestamp']))
                                match = True
                                break
                        if match:
                            del finish_events[j]
                    if match:
                        ordered_event_log.extend(temp_trace)
        return ordered_event_log

    def get_mxml_events_data(self, filename, start_timeformat, end_timeformat):
        """read and parse all the events information from a MXML file"""
        temp_data = list()
        tree = ET.parse(filename)
        root = tree.getroot()
        process = root.find('Process')
        procInstas = process.findall('ProcessInstance')
        i = 0
        for procIns in tqdm(procInstas, unit="Process Instance", desc="Reading Log Traces"):
            caseid = procIns.get('id')
            complete_timestamp = ''
            auditTrail = procIns.findall('AuditTrailEntry')
            for trail in auditTrail:
                task = ''
                user = ''
                event_type = ''
                type_task = ''
                timestamp = ''
                attributes = trail.find('Data').findall('Attribute')
                for attr in attributes:
                    if (attr.get('name') == 'concept:name'):
                        task = attr.text
                    if (attr.get('name') == 'lifecycle:transition'):
                        event_type = attr.text
                    if (attr.get('name') == 'org:resource'):
                        user = attr.text
                    if (attr.get('name') == 'type_task'):
                        type_task = attr.text
                work_flow_ele = trail.find('WorkflowModelElement').text
                event_type = trail.find('EventType').text
                timestamp = trail.find('Timestamp').text
                originator = trail.find('Originator').text
                timestamp = datetime.datetime.strptime(trail.find('Timestamp').text[:-6], start_timeformat)
                temp_data.append(
                    dict(caseid=caseid, task=task, event_type=event_type, user=user, start_timestamp=timestamp,
                         end_timestamp=timestamp))

            i += 1
        raw_data = temp_data
        temp_data = self.reorder_mxml(temp_data)
        return temp_data, raw_data

    def reorder_mxml(self, temp_data):
        """this method joints the duplicated events on the .mxml log"""
        data = list()
        start_events = list(filter(lambda x: x['event_type'] == 'start', temp_data))
        finish_events = list(filter(lambda x: x['event_type'] == 'complete', temp_data))
        for x, y in zip(start_events, finish_events):
            data.append(dict(caseid=x['caseid'], task=x['task'], event_type=x['event_type'],
                             user=x['user'], start_timestamp=x['start_timestamp'], end_timestamp=y['start_timestamp']))
        return data

    def get_csv_events_data(self, log_columns_numbers, start_timeformat, end_timeformat):
        """reads and parse all the events information from a csv file"""
        flength = file_size(self.input)
        i = 0
        temp_data = list()
        with open(self.input, 'r') as csvfile:
            filereader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(filereader, None)  # skip the headers
            for row in tqdm(filereader, unit="row", desc="Reading CSV Event Data"):
                timestamp = ''
                complete_timestamp = ''
                if row[log_columns_numbers[1]] != 'End':
                    timestamp = datetime.datetime.strptime(row[log_columns_numbers[4]], start_timeformat)
                    complete_timestamp = datetime.datetime.strptime(row[log_columns_numbers[5]], end_timeformat)
                temp_data.append(dict(caseid=row[log_columns_numbers[0]], task=row[log_columns_numbers[1]],
                                      event_type=row[log_columns_numbers[2]], user=row[log_columns_numbers[3]],
                                      start_timestamp=timestamp, end_timestamp=complete_timestamp))
                i += 1
        return temp_data, temp_data

    # TODO manejo de excepciones
    def find_first_task(self):
        """finds the first task"""
        cases = list()
        [cases.append(c['caseid']) for c in self.data]
        cases = sorted(list(set(cases)))
        first_task_names = list()
        for case in cases:
            trace = sorted(list(filter(lambda x: (x['caseid'] == case), self.data)), key=itemgetter('start_timestamp'))
            first_task_names.append(trace[0]['task'])
        first_task_names = list(set(first_task_names))
        return first_task_names

    def get_traces(self):
        """returns the data splitted by caseid and ordered by start_timestamp"""
        cases = list()
        for c in self.data: cases.append(c['caseid'])
        cases = sorted(list(set(cases)))
        traces = list()
        for case in cases:
            # trace = sorted(list(filter(lambda x: (x['caseid'] == case), self.data)), key=itemgetter('start_timestamp'))
            trace = list(filter(lambda x: (x['caseid'] == case), self.data))
            traces.append(trace)
        return traces

    def get_raw_traces(self):
        """returns the raw data splitted by caseid and ordered by start_timestamp"""
        cases = list()
        for c in self.raw_data: cases.append(c['caseid'])
        cases = sorted(list(set(cases)))
        traces = list()
        for case in cases:
            trace = sorted(list(filter(lambda x: (x['caseid'] == case), self.raw_data)),
                           key=itemgetter('start_timestamp'))
            traces.append(trace)
        return traces

    def read_resource_task(self, task, roles):
        """returns the resource that performs a task"""
        filtered_list = list(filter(lambda x: x['task'] == task, self.data))
        role_assignment = list()
        for task in filtered_list:
            for role in roles:
                for member in role['members']:
                    if task['user'] == member:
                        role_assignment.append(role['role'])
        return max(role_assignment)

    def set_data(self, data):
        """seting method for the data attribute"""
        self.data = data

# Data Reader specific utils

import scipy
from scipy.stats import pearsonr
import networkx as nx
import matplotlib.pyplot as plt
from operator import itemgetter
import random


# == support
def random_color(size):
    number_of_colors = size
    color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]
    return color


def find_index(dictionary, value):
    finish = False
    i = 0
    resp = -1
    while i < len(dictionary) and not finish:
        if dictionary[i]['data'] == value:
            resp = dictionary[i]['index']
            finish = True
        i += 1
    return resp


def det_freq_matrix(unique, dictionary):
    freq_matrix = list()
    for u in unique:
        freq = 0
        for d in dictionary:
            if u == d:
                freq += 1
        freq_matrix.append(dict(task=u[0], user=u[1], freq=freq))
    return freq_matrix


def build_profile(users, freq_matrix, prof_size):
    profiles = list()
    for user in users:
        exec_tasks = list(filter(lambda x: x['user'] == user['index'], freq_matrix))
        profile = [0, ] * prof_size
        for exec_task in exec_tasks:
            profile[exec_task['task']] = exec_task['freq']
        profiles.append(dict(user=user['index'], profile=profile))
    return profiles


def det_correlation_matrix(profiles):
    correlation_matrix = list()
    for profile_x in profiles:
        for profile_y in profiles:
            x = scipy.array(profile_x['profile'])
            y = scipy.array(profile_y['profile'])
            r_row, p_value = pearsonr(x, y)
            correlation_matrix.append(dict(x=profile_x['user'], y=profile_y['user'], distance=r_row))
    return correlation_matrix


# =============================================================================
# def graph_network(g):
#     pos = nx.spring_layout(g, k=0.5,scale=10) nx.draw_networkx(g,pos,node_size=200,with_labels=True,font_size=11,
#     font_color='#A0CBE2')
#     edge_labels=dict([((u,v,),round(d['weight'],2)) for u,v,d in g.edges(data=True)])
#     nx.draw_networkx_edge_labels(g,pos,edge_labels=edge_labels)
#     plt.draw()
#     plt.show()
#
# =============================================================================
def graph_network(g, sub_graphs):
    # IDEA se debe calcular el centroide de los clusters....pos es un diccionario de posiciones y el centroide es el promedio de los puntos x y y
    # despues se debe determinar el punto mas lejano del centroide y ese sera el radio y con esos datos pintar un circulo con patches
    pos = nx.spring_layout(g, k=0.5, scale=10)
    color = random_color(len(sub_graphs))
    for i in range(0, len(sub_graphs)):
        subgraph = sub_graphs[i]
        nx.draw_networkx_nodes(g, pos, nodelist=list(subgraph), node_color=color[i], node_size=200, alpha=0.8)
        nx.draw_networkx_edges(g, pos, width=1.0, alpha=0.5)
        nx.draw_networkx_edges(g, pos, edgelist=subgraph.edges, width=8, alpha=0.5, edge_color=color[i])
    plt.draw()
    plt.show()  # display


def connected_component_subgraphs(G):
    for c in nx.connected_components(G):
        yield G.subgraph(c)


def role_definition(sub_graphs, users):
    records = list()
    for i in range(0, len(sub_graphs)):
        users_names = list()
        for user in sub_graphs[i]:
            users_names.append(list(filter(lambda x: x['index'] == user, users))[0]['data'])
        records.append(dict(role='Role ' + str(i + 1), quantity=len(sub_graphs[i]), members=users_names))
    # Sort roles by number of resources
    records = sorted(records, key=itemgetter('quantity'), reverse=True)
    for i in range(0, len(records)):
        records[i]['role'] = 'Role ' + str(i + 1)
    resource_table = list()
    for record in records:
        for member in record['members']:
            resource_table.append(dict(role=record['role'], resource=member))
    return records, resource_table


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

    print(log_parameter)

    # Construct rename mapping to fit the pm4py xes standards from the log parameters
    xes_column_remap = {
        log_parameter["case_id_key"]: "case:" + xes_const.DEFAULT_NAME_KEY,  # "case:concept:name"
        log_parameter["activity_key"]: xes_const.DEFAULT_NAME_KEY,  # "concept:name"
        log_parameter["timestamp_key"]: xes_const.DEFAULT_TIMESTAMP_KEY,  # "time:timestamp"
    }
    if "resource_key" in log_parameter:
        xes_column_remap[log_parameter["resource_key"]] = "org:resource"

    log_df.rename(xes_column_remap, axis=1, inplace=True)

    return log_df


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
    print(log_parameter)

    df = ensure_xes_standard_naming(df, log_parameter)

    df = pm4py.format_dataframe(df)

    try:
        df['time:timestamp'] = df['time:timestamp'].apply(lambda x: datetime.datetime.strptime(x, "%Y/%m/%d %H:%M:%S.%f") if isinstance(x, str) else x)
    except ValueError as e:
        df['time:timestamp'] = df['time:timestamp'].apply(lambda x: datetime.datetime.fromisoformat(x) if isinstance(x, str) else x)


    event_log = pm4py.convert_to_event_log(df)
    if not xes_path:
        xes_path = csv_path.with_suffix(".xes")
    pm4py.write_xes(event_log, str(xes_path))

    return xes_path
