import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from keras import backend as K
from keras.models import Model
from keras.preprocessing import sequence
from keras.utils.data_utils import Sequence
from keras.regularizers import l2
from keras.constraints import non_neg, Constraint
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve


from keras.layers import Input, Concatenate, Flatten
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.optimizers import Nadam, Adam, SGD, Adagrad
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization

import keras
print(keras.__version__)


# -*- coding: utf-8 -*-
from sys import stdout
import numpy as np
import datetime
import os
import csv
import uuid
import json
import platform as pl


def folder_id():
    return datetime.datetime.today().strftime('%Y%m%d_%H%M%S%f')
#generate unique bimp element ids
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

#printing formated float
def ffloat(num, dec):
    return float("{0:.2f}".format(np.round(num,decimals=dec)))

#transform a string into date object
#def get_time_obj(date, timeformat):
#    date_modified = datetime.datetime.strptime(date,timeformat)
#    return date_modified


#print debuging csv file
def create_csv_file(index, output_file, mode='w'):
    with open(output_file, mode) as f:
        for element in index:
            w = csv.DictWriter(f, element.keys())
            w.writerow(element)
        f.close()

def create_csv_file_header(index, output_file, mode='w'):
    with open(output_file, mode, newline='') as f:
        fieldnames = index[0].keys()
        w = csv.DictWriter(f, fieldnames)
        w.writeheader()
        for element in index:
            w.writerow(element)
        f.close()

def create_json(dictionary, output_file):
    with open(output_file, 'w') as f:
         f.write(json.dumps(dictionary))
         
# rounding lists values preserving the sum values
def round_preserve(l,expected_sum):
    actual_sum = sum(l)
    difference = round(expected_sum - actual_sum,2)
    if difference > 0.00:
        idx= l.index(min(l))
    else:
        idx= l.index(max(l))
    l[idx] +=difference
    return l

## added code to save figure
def plot_history( plt, figure_name, path, save_fig=True ):
  
  fig_name = figure_name + ".png"
  full_path = path + fig_name
  if save_fig:
    plt.savefig(full_path, dpi=300)
 


# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 16:53:16 2018
This module contains support functions specifically created to manipulate 
Event logs in pandas dataframe format
@author: Manuel Camargo
"""
import numpy as np
import pandas as pd

# =============================================================================
# Split an event log dataframe to peform split-validation 
# =============================================================================
def split_train_test(df, percentage):
    cases = df.caseid.unique()
    num_test_cases = int(np.round(len(cases)*percentage))
    test_cases = cases[:num_test_cases]
    train_cases = cases[num_test_cases:]
    df_train, df_test = pd.DataFrame(), pd.DataFrame()
    for case in train_cases:
        df_train = df_train.append(df[df.caseid==case]) 
    df_train = df_train.sort_values('start_timestamp', ascending=True).reset_index(drop=True)
 
    for case in test_cases:
        df_test = df_test.append(df[df.caseid==case]) 
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
   df['dur'] = (df.end_timestamp-df.start_timestamp).apply(get_seconds)
   # Time between activities per trace
   df['tbtw'] = 0
   # Multitasking time
   cases = df.caseid.unique()
   for case in cases:
       trace = df[df.caseid==case].sort_values('start_timestamp', ascending=True)
       for i in range(1,len(trace)):
           row_num = trace.iloc[i].name
           tbtw = (trace.iloc[i].start_timestamp - trace.iloc[i - 1].end_timestamp).seconds
           df.iloc[row_num,df.columns.get_loc('tbtw')] = tbtw
   return df, cases

# =============================================================================
# Standardization
# =============================================================================

def max_min_std(df, serie):
    max_value, min_value = np.max(df[serie]), np.min(df[serie])
    std = lambda x: (x[serie] - min_value) / (max_value - min_value)
    df[serie+'_norm']=df.apply(std,axis=1)
    return df, max_value, min_value


def max_std(df, serie):
    max_value, min_value = np.max(df[serie]), np.min(df[serie])
    std = lambda x: x[serie] / max_value
    df[serie+'_norm']=df.apply(std,axis=1)
    return df, max_value, min_value

def max_min_de_std(val, max_value, min_value):
    true_value = (val * (max_value - min_value)) + min_value
    return true_value

def max_de_std(val, max_value, min_value):
    true_value = val * max_value 
    return true_value


# -*- coding: utf-8 -*-
import scipy
from scipy.stats import pearsonr
import networkx as nx
import matplotlib.pyplot as plt
#from support_modules import support as sup
from operator import itemgetter
import random


# == support
def random_color(size):
    number_of_colors = size
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]
    return color

def find_index(dictionary, value):
    finish = False
    i = 0
    resp = -1
    while i<len(dictionary) and not finish:
        if dictionary[i]['data']==value:
            resp = dictionary[i]['index']
            finish = True
        i+=1
    return resp

def det_freq_matrix(unique, dictionary):
    freq_matrix = list()
    for u in unique:
        freq = 0
        for d in dictionary:
            if u == d:
                freq += 1
        freq_matrix.append(dict(task=u[0],user=u[1],freq=freq))
    return freq_matrix

def build_profile(users,freq_matrix,prof_size):
    profiles=list()
    for user in users:
        exec_tasks = list(filter(lambda x: x['user']==user['index'],freq_matrix))
        profile = [0,] * prof_size
        for exec_task in exec_tasks:
            profile[exec_task['task']]=exec_task['freq']
        profiles.append(dict(user=user['index'],profile=profile))
    return profiles

def det_correlation_matrix(profiles):
    correlation_matrix = list()
    for profile_x in profiles:
        for profile_y in profiles:
            x = scipy.array(profile_x['profile'])
            y = scipy.array(profile_y['profile'])
            r_row, p_value = pearsonr(x, y)
            correlation_matrix.append(dict(x=profile_x['user'],y=profile_y['user'],distance=r_row))
    return correlation_matrix

# =============================================================================
# def graph_network(g):
#     pos = nx.spring_layout(g, k=0.5,scale=10)
#     nx.draw_networkx(g,pos,node_size=200,with_labels=True,font_size=11, font_color='#A0CBE2')
#     edge_labels=dict([((u,v,),round(d['weight'],2)) for u,v,d in g.edges(data=True)])
#     nx.draw_networkx_edge_labels(g,pos,edge_labels=edge_labels)
#     plt.draw()
#     plt.show()
# 
# =============================================================================
def graph_network(g, sub_graphs):
    #IDEA se debe calcular el centroide de los clusters....pos es un diccionario de posiciones y el centroide es el promedio de los puntos x y y
    #despues se debe determinar el punto mas lejano del centroide y ese sera el radio y con esos datos pintar un circulo con patches
    pos = nx.spring_layout(g, k=0.5,scale=10)
    color = random_color(len(sub_graphs))
    for i in range(0,len(sub_graphs)):
        subgraph = sub_graphs[i]
        nx.draw_networkx_nodes(g,pos, nodelist=list(subgraph), node_color=color[i], node_size=200, alpha=0.8)
        nx.draw_networkx_edges(g,pos,width=1.0,alpha=0.5)
        nx.draw_networkx_edges(g,pos, edgelist=subgraph.edges, width=8,alpha=0.5,edge_color=color[i])
    plt.draw()
    plt.show() # display

def connected_component_subgraphs(G):
    for c in nx.connected_components(G):
        yield G.subgraph(c)

def role_definition(sub_graphs,users):
    records= list()
    for i in range(0,len(sub_graphs)):
        users_names = list()
        for user in sub_graphs[i]:
            users_names.append(list(filter(lambda x: x['index']==user,users))[0]['data'])
        records.append(dict(role='Role '+ str(i + 1),quantity =len(sub_graphs[i]),members=users_names))
    #Sort roles by number of resources
    records = sorted(records, key=itemgetter('quantity'), reverse=True)
    for i in range(0,len(records)):
        records[i]['role']='Role '+ str(i + 1)
    resource_table = list()
    for record in records:
        for member in record['members']:
            resource_table.append(dict(role=record['role'], resource=member))
    return records, resource_table
# --kernel--

def role_discovery(data, drawing, sim_percentage):
    tasks = list(set(list(map(lambda x: x[0], data))))
    try:
        tasks.remove('Start')
    except Exception:
    	pass
    tasks = [dict(index=i,data=tasks[i]) for i in range(0,len(tasks))]
    users = list(set(list(map(lambda x: x[1], data))))
    try:
        users.remove('Start')
    except Exception:
    	pass
    users = [dict(index=i,data=users[i]) for i in range(0,len(users))]
    data_transform = list(map(lambda x: [find_index(tasks, x[0]),find_index(users, x[1])], data ))
    unique = list(set(tuple(i) for i in data_transform))
    unique = [list(i) for i in unique]
    # [print(uni) for uni in users]
    # building of a task-size profile of task execution per resource
    profiles = build_profile(users,det_freq_matrix(unique,data_transform),len(tasks))
    print_performed_task('Analysing resource pool ')
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
        if relation['distance'] > sim_percentage and relation['x']!=relation['y'] :
            g.add_edge(relation['x'],relation['y'],weight=relation['distance'])
#    sup.print_progress(((60 / 100)* 100),'Analysing resource pool ')
    # extraction of fully conected subgraphs as roles
    sub_graphs = list(connected_component_subgraphs(g))
#    sup.print_progress(((80 / 100)* 100),'Analysing resource pool ')
    # role definition from graph
    roles = role_definition(sub_graphs,users)
    # plot creation (optional)
    if drawing == True:
        graph_network(g, sub_graphs)
#    sup.print_progress(((100 / 100)* 100),'Analysing resource pool ')
    print_done_task()
    return roles

def read_roles_from_columns(raw_data, filtered_data, separator):
	records = list()
	role_list= list()
	pool_list= list()
	raw_splited= list()
	for row in raw_data:
		temp = row.split(separator)
		if temp[0] != 'End':
			raw_splited.append(dict(role=temp[1],resource=temp[0]))
	for row in filtered_data:
		temp = row.split(separator)
		if temp[0] != 'End':
			pool_list.append(dict(role=temp[1],resource=temp[0]))
			role_list.append(temp[1])
	role_list = list(set(role_list))
	for role in role_list:
		members = list(filter(lambda person: person['role'] == role, pool_list))
		members = list(map(lambda x: x['resource'],members))
		quantity = len(members)
		#freq = len(list(filter(lambda person: person['role'] == role, raw_splited)))
		records.append(dict(role=role,quantity =quantity,members=members))
	return records

def read_resource_pool(log, separator=None, drawing=False, sim_percentage=0.7):
    if separator == None:
        filtered_list = list()
        for row in log.data:
            if row['task'] != 'End' and row['user'] != 'AUTO':
                filtered_list.append([row['task'],row['user']])
        return role_discovery(filtered_list, drawing, sim_percentage)
    else:
        raw_list = list()
        filtered_list = list()
        for row in log.data:
            raw_list.append(row['user'])
        filtered_list = list(set(raw_list))
        return read_roles_from_columns(raw_list, filtered_list, separator)


# -*- coding: utf-8 -*-
import csv
import datetime
import xml.etree.ElementTree as ET
import gzip
import zipfile as zf
import os
from operator import itemgetter

class LogReader(object):
    """
	This class reads and parse the elements of a given process log in format .xes or .csv
	"""

    def __init__(self, input, start_timeformat, end_timeformat, log_columns_numbers=[], ns_include=True, one_timestamp=False):
        """constructor"""
        self.input = input
        self.data, self.raw_data = self.load_data_from_file(log_columns_numbers, start_timeformat, end_timeformat, ns_include, one_timestamp)


    # Support Method
    def define_ftype(self):
        filename, file_extension = os.path.splitext(self.input)
        if file_extension == '.xes' or file_extension == '.csv' or file_extension == '.mxml' :
             filename = filename + file_extension
             file_extension = file_extension
        elif file_extension == '.gz':
            outFileName = filename
            filename, file_extension = self.decompress_file_gzip(self.input, outFileName)
        elif file_extension=='.zip':
            filename,file_extension = self.decompress_file_zip(self.input, filename)
        elif not (file_extension == '.xes' or file_extension == '.csv' or file_extension == '.mxml'):
            raise IOError('file type not supported')
        return filename,file_extension

    # Decompress .gz files
    def decompress_file_gzip(self,filename, outFileName):
        inFile = gzip.open(filename, 'rb')
        outFile = open(outFileName,'wb')
        outFile.write(inFile.read())
        inFile.close()
        outFile.close()
        _, fileExtension = os.path.splitext(outFileName)
        return outFileName,fileExtension

    # Decompress .zip files
    def decompress_file_zip(self, filename, outfilename):
        with zf.ZipFile(filename,"r") as zip_ref:
            zip_ref.extractall("../inputs/")
        _, fileExtension = os.path.splitext(outfilename)
        return outfilename, fileExtension

    # Reading methods
    def load_data_from_file(self, log_columns_numbers, start_timeformat, end_timeformat, ns_include, one_timestamp):
        """reads all the data from the log depending the extension of the file"""
        temp_data = list()
        filename, file_extension = self.define_ftype()
        if file_extension == '.xes':
            temp_data, raw_data = self.get_xes_events_data(filename,start_timeformat, end_timeformat, ns_include, one_timestamp)
        elif file_extension == '.csv':
            temp_data, raw_data = self.get_csv_events_data(log_columns_numbers, start_timeformat, end_timeformat)
        elif file_extension == '.mxml':
            temp_data, raw_data = self.get_mxml_events_data(filename,start_timeformat, end_timeformat)
        return temp_data, raw_data

    def get_xes_events_data(self, filename,start_timeformat, end_timeformat, ns_include, one_timestamp):
        """reads and parse all the events information from a xes file"""
        temp_data = list()
        tree = ET.parse(filename)
        root = tree.getroot()
        if ns_include:
            #TODO revisar como poder cargar el mane space de forma automatica del root
            ns = {'xes': root.tag.split('}')[0].strip('{')}
            tags = dict(trace='xes:trace',string='xes:string',event='xes:event',date='xes:date')
        else:
            ns = {'xes':''}
            tags = dict(trace='trace',string='string',event='event',date='date')
        traces = root.findall(tags['trace'], ns)
        i = 0
        print_performed_task('Reading log traces ')
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
                            timestamp = datetime.datetime.strptime(timestamp[:-6], start_timeformat)
                        except ValueError:
                            timestamp = datetime.datetime.strptime(timestamp, start_timeformat)
                if not (task == '0' or task == '-1'):
                    temp_data.append(
                        dict(caseid=caseid, task=task, event_type=event_type, user=user, start_timestamp=timestamp,
                             end_timestamp=complete_timestamp))
            i += 1
        raw_data = temp_data
        temp_data = self.reorder_xes(temp_data, one_timestamp)
        print_done_task()
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
                start_events = sorted(list(filter(lambda x: x['event_type'] == 'start' and x['caseid'] == case, events)), key=lambda x:x['start_timestamp'])
                finish_events = sorted(list(filter(lambda x: x['event_type'] == 'complete' and x['caseid'] == case, events)), key=lambda x:x['start_timestamp'])
                if len(start_events) == len(finish_events):
                    temp_trace = list()
                    for i, _ in enumerate(start_events):
                        match = False
                        for j, _ in enumerate(finish_events):
                            if start_events[i]['task'] == finish_events[j]['task']:
                                temp_trace.append(dict(caseid=case, task=start_events[i]['task'], event_type=start_events[i]['task'],
                                     user=start_events[i]['user'], start_timestamp=start_events[i]['start_timestamp'], end_timestamp=finish_events[j]['start_timestamp']))
                                match = True
                                break
                        if match:
                            del finish_events[j]
                    if match:
                        ordered_event_log.extend(temp_trace)
        return ordered_event_log

    def get_mxml_events_data(self, filename,start_timeformat, end_timeformat):
        """read and parse all the events information from a MXML file"""
        temp_data = list()
        tree = ET.parse(filename)
        root = tree.getroot()
        process = root.find('Process')
        procInstas = process.findall('ProcessInstance')
        i = 0
        for procIns in procInstas:
            print_progress(((i / (len(procInstas) - 1)) * 100), 'Reading log traces ')
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
        print_done_task()
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
            for row in filereader:
                print_progress(((i / (flength - 1)) * 100), 'Reading log traces ')
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
            trace = sorted(list(filter(lambda x: (x['caseid'] == case), self.raw_data)), key=itemgetter('start_timestamp'))
            traces.append(trace)
        return traces

    def read_resource_task(self,task,roles):
        """returns the resource that performs a task"""
        filtered_list = list(filter(lambda x: x['task']==task, self.data))
        role_assignment = list()
        for task in filtered_list:
            for role in roles:
                for member in role['members']:
                    if task['user']==member:
                        role_assignment.append(role['role'])
        return max(role_assignment)

    def set_data(self,data):
        """seting method for the data attribute"""
        self.data = data


import keras.layers as L
from keras import backend as K
from keras.layers import Embedding

from keras.layers import Lambda, dot, Activation, concatenate, Dense




def training_model_temporal_variable(vec, ac_weights, rl_weights, output_folder, args):

  
  dropout_input = 0.01
  dropout_context=0.30
  lstm_size_alpha=args['l_size']
  lstm_size_beta=args['l_size']
  print("Training prefix and variable attention model")

  l2reg=0.0001
  allow_negative=False
  incl_time = True 
  incl_res = True
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
  alpha = L.Bidirectional(L.CuDNNLSTM(lstm_size_alpha, return_sequences=True),
                                    name='alpha')
  beta = L.Bidirectional(L.CuDNNLSTM(lstm_size_beta, return_sequences=True),
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

def training_model_with_time_prediction(vec, ac_weights, rl_weights, output_folder, args):


  dropout_input = 0.01
  dropout_context=0.35
  lstm_size_alpha=args['l_size']
  lstm_size_beta=args['l_size']
  print("Training activity, time and role with attention")
  l2reg=0.0005
  allow_negative=False
  incl_time = True 
  incl_res = True
  ac_input = Input(shape=(vec['prefixes']['x_ac_inp'].shape[1], ), name='ac_input')
  rl_input = Input(shape=(vec['prefixes']['x_rl_inp'].shape[1], ), name='rl_input')
  t_input = Input(shape=(vec['prefixes']['xt_inp'].shape[1], 1), name='t_input')

 
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
      #input_list.append(t_input)
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
 
  #batch1 = BatchNormalization()(contexts)


  act_output = Dense(ac_weights.shape[0],
                       activation='softmax',
                       kernel_initializer='glorot_uniform',
                       name='act_output')(contexts)

  rl_output = Dense(rl_weights.shape[0],
                       activation='softmax',
                       kernel_initializer='glorot_uniform',
                       name='rl_output')(contexts)


  #t1_output = Dense(10, kernel_initializer='glorot_uniform',
  #                     name='tin_output')(contexts)



  t_output = Dense(1, kernel_initializer='glorot_uniform',
                       name='t_output')(contexts)
 
  model = Model(inputs=[ac_input, rl_input, t_input], outputs=[act_output, rl_output, t_output])

  if args['optim'] == 'Nadam':
      opt = Nadam(lr=0.001, beta_1=0.9, beta_2=0.999,
                    epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
  elif args['optim'] == 'Adam':
      opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
                   epsilon=None, decay=0.0, amsgrad=False)
  elif args['optim'] == 'SGD':
      opt = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
  elif args['optim'] == 'Adagrad':
      opt = Adagrad(lr=0.01, epsilon=None, decay=0.0)

  model.compile(loss={'t_output':'mae','act_output':'categorical_crossentropy', 'rl_output':'categorical_crossentropy'}, 
                optimizer=opt, metrics=['accuracy','mae'])
    
  model.summary()
    
  early_stopping = EarlyStopping(monitor='val_loss',verbose=1, patience=42)
#
#    # Output file
  output_file_path = os.path.join(output_folder,
                                    'models/model_rd_' + str(args['l_size']) +
                                    ' ' + args['optim'] +  args['log_name'] +
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

  model.fit({'ac_input':vec['prefixes']['x_ac_inp'],
               'rl_input':vec['prefixes']['x_rl_inp'],
               't_input':vec['prefixes']['xt_inp']},
              {'act_output':vec['next_evt']['y_ac_inp'],
               'rl_output':vec['next_evt']['y_rl_inp'],
               't_output':vec['next_evt']['yt_inp']},
              validation_split=0.15,
              verbose=0,
              callbacks=[early_stopping, model_checkpoint, lr_reducer],
              batch_size=50,
              epochs=100)
  return model

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 21:23:55 2018

@author: Manuel Camargo
"""
import os
import csv
import math
import itertools

import keras.utils as ku

import pandas as pd
import numpy as np
import pickle

from nltk.util import ngrams

def training_model(timeformat, args, no_loops=False):
    """Main method of the training module.
    Args:
        timeformat (str): event-log date-time format.
        args (dict): parameters for training the network.
        no_loops (boolean): remove loops fom the event-log (optional).
    """
    parameters = dict()
    # read the logfile
    log = LogReader( args['file_name'],
                       timeformat, timeformat, one_timestamp=True)
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
    rl_weights =  ku.to_categorical(sorted(index_rl.keys()), len(rl_index))
    print('RL_WEIGHTS', rl_weights)


    # Calculate relative times
    log_df = add_calculated_features(log_df, ac_index, rl_index)
    # Split validation datasets
    log_df_train, log_df_test = split_train_test(log_df, 0.3) # 70%/30%
    # Input vectorization
    vec = vectorization(log_df_train, ac_index, rl_index, args)
    #print(vec['prefixes']['x_ac_inp'])

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
                                            args['log_name']+'model_parameters.json'))

    #pickle.dump(vec, open( os.path.join(output_folder,
    #                                        'parameters',
    #                                    args['log_name']+'train_vec.pkl'), "wb"))


    create_csv_file_header(log_df_test.to_dict('records'),
                               os.path.join(output_folder,
                                            'parameters',
                                            args['log_name']+'test_log.csv'))




    if(args['task']=='prefix_attn'):
        model = training_model_temporal(vec, ac_weights, rl_weights, output_folder, args)
    elif(args['task']=='full_attn'):
        model = training_model_temporal_variable(vec, ac_weights, rl_weights, output_folder, args)
    else:
        model = training_model_with_time_prediction(vec, ac_weights, rl_weights, output_folder, args)

    #elif args['model_type'] == 'shared_cat':
    #    training_model_sharedcat(vec, ac_weights, rl_weights, output_folder, args)


# =============================================================================
# Pre-processing: n-gram vectorization
# =============================================================================
def vectorization(log_df, ac_index, rl_index, args):
    """Example function with types documented in the docstring.
    Args:
        log_df (dataframe): event log data.
        ac_index (dict): index of activities.
        rl_index (dict): index of roles.
        args (dict): parameters for training the network
    Returns:
        dict: Dictionary that contains all the LSTM inputs.
    """
    if args['norm_method'] == 'max':
        mean_tbtw = np.mean(log_df.tbtw)
        std_tbtw = np.std(log_df.tbtw)
        norm = lambda x: (x['tbtw']-mean_tbtw)/std_tbtw
        log_df['tbtw_norm'] = log_df.apply(norm, axis=1)
        log_df = reformat_events(log_df, ac_index, rl_index)
    elif args['norm_method'] == 'lognorm':
        logit = lambda x: math.log1p(x['tbtw'])
        log_df['tbtw_log'] = log_df.apply(logit, axis=1)
        mean_tbtw = np.mean(log_df.tbtw_log)
        std_tbtw=np.std(log_df.tbtw_log)
        norm = lambda x: (x['tbtw_log']-mean_tbtw)/std_tbtw
        log_df['tbtw_norm'] = log_df.apply(norm, axis=1)
        log_df = reformat_events(log_df, ac_index, rl_index)

    vec = {'prefixes':dict(), 'next_evt':dict(), 'mean_tbtw':mean_tbtw, 'std_tbtw':std_tbtw}
    # n-gram definition
    for i, _ in enumerate(log_df):
        ac_n_grams = list(ngrams(log_df[i]['ac_order'], args['n_size'],
                                 pad_left=True, left_pad_symbol=0))
        rl_n_grams = list(ngrams(log_df[i]['rl_order'], args['n_size'],
                                 pad_left=True, left_pad_symbol=0))
        tn_grams = list(ngrams(log_df[i]['tbtw'], args['n_size'],
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
        for j in range(st_idx, len(ac_n_grams)-1):
            vec['prefixes']['x_ac_inp'] = np.concatenate((vec['prefixes']['x_ac_inp'],
                                                          np.array([ac_n_grams[j]])), axis=0)
            vec['prefixes']['x_rl_inp'] = np.concatenate((vec['prefixes']['x_rl_inp'],
                                                          np.array([rl_n_grams[j]])), axis=0)
            vec['prefixes']['xt_inp'] = np.concatenate((vec['prefixes']['xt_inp'],
                                                        np.array([tn_grams[j]])), axis=0)
            vec['next_evt']['y_ac_inp'] = np.append(vec['next_evt']['y_ac_inp'],
                                                    np.array(ac_n_grams[j+1][-1]))
            vec['next_evt']['y_rl_inp'] = np.append(vec['next_evt']['y_rl_inp'],
                                                    np.array(rl_n_grams[j+1][-1]))
            vec['next_evt']['yt_inp'] = np.append(vec['next_evt']['yt_inp'],
                                                  np.array(tn_grams[j+1][-1]))

    vec['prefixes']['xt_inp'] = vec['prefixes']['xt_inp'].reshape(
        (vec['prefixes']['xt_inp'].shape[0],
         vec['prefixes']['xt_inp'].shape[1], 1))



    #print(vec['prefixes']['x_ac_inp'])
    vec['next_evt']['y_ac_inp'] = ku.to_categorical(vec['next_evt']['y_ac_inp'],
                                                    num_classes=len(ac_index))

    vec['next_evt']['y_rl_inp'] = ku.to_categorical(vec['next_evt']['y_rl_inp'],
                                                    num_classes=len(rl_index))

    return vec

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

    log_df = log_df.to_dict('records')

    log_df = sorted(log_df, key=lambda x: (x['caseid'], x['end_timestamp']))
    for _, group in itertools.groupby(log_df, key=lambda x: x['caseid']):
        trace = list(group)
        for i, _ in enumerate(trace):
            if i != 0:
                trace[i]['tbtw'] = (trace[i]['end_timestamp'] -
                                    trace[i-1]['end_timestamp']).total_seconds()

    return pd.DataFrame.from_records(log_df)

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



MY_WORKSPACE_DIR = "/content/drive/My Drive/BPIC_Data/"


#data_train_df = pd.read_pickle(MY_WORKSPACE_DIR +"data_train.pkl")
 
import sys
import getopt


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 


def catch_parameter(opt):
    """Change the captured parameters names"""
    switch = {'-h':'help', '-i':'imp', '-l':'lstm_act',
              '-d':'dense_act', '-n':'norm_method', '-f':'folder',
              '-m':'model_file', '-t':'model_type', '-a':'activity',
              '-e':'file_name', '-b':'n_size', '-c':'l_size', '-o':'optim'}
    try:
        return switch[opt]
    except:
        raise Exception('Invalid option ' + opt)

# --setup--
def main():
    """Main aplication method"""
    timeformat = '%Y-%m-%dT%H:%M:%S.%f'
    parameters = dict()
#   Parameters setting manual fixed or catched by console for batch operations
   
    
    parameters['folder'] = "/content/drive/My Drive/BPIC_Data/output_files/"
#       Specific model training parameters
    parameters['imp'] = 1 # keras lstm implementation 1 cpu, 2 gpu
    parameters['lstm_act'] = None # optimization function see keras doc
    parameters['dense_act'] = None # optimization function see keras doc
    parameters['optim'] = 'Adagrad' # optimization function see keras doc
    parameters['norm_method'] = 'lognorm' # max, lognorm
                # Model types --> specialized, concatenated, shared_cat, joint, shared
    parameters['model_type'] = 'shared_cat'
    parameters['l_size'] = 50 # LSTM layer sizes
#       Generation parameters
    parameters['folder'] = "/content/drive/My Drive/BPIC_Data/output_files/"
    parameters['file_name'] = MY_WORKSPACE_DIR + 'BPI_Challenge_2012.xes.gz' #'BPI_2012_W_complete.xes.gz'
    #parameters['model_file'] = 'model_rd_100 Nadam_02-0.90.h5'
    parameters['n_size'] = 15 # n-gram size

    parameters['log_name'] = 'bpic2012_15_lstm_sufftime'

    parameters['task']='prefix_attn111'
    
    training_model(timeformat, parameters)
    
 


if __name__ == "__main__":
    main()

###########
# READDED #
###########
measurements = None
if measurements:
    if os.path.exists(os.path.join(output_route, file_name)):
        create_csv_file(measurements, os.path.join(output_route, file_name), mode='a')
    else:
        create_csv_file_header(measurements, os.path.join(output_route,file_name))
