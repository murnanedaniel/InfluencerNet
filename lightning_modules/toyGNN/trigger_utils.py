# System imports
import sys
import os

# External imports
import numpy as np

import torch
from torch_geometric.data import Data
import scipy as sp

import warnings
import random
from typing import Type
import functools
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.append("../../..")
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_dataset(input_dir="/global/cfs/cdirs/m3443/data/TrackLRP/toy_dataset_v1", datatype_split=None, attention_cut=None, **kwargs):
    if datatype_split is None:
        datatype_split = [500, 100, 10]
        
    subdirs = ["train", "val", "test"]
    return [load_toy_dataset(os.path.join(input_dir, subdir), num_events, attention_cut, **kwargs) for num_events, subdir in zip(datatype_split, subdirs)]

def load_toy_dataset(input_dir, num_events, attention_cut, **kwargs):

    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    random.shuffle(files)
    files = files[:num_events]

    dataset = [ torch.load(f) for f in files ]
    if attention_cut is not None:
        for data in dataset:
            data.edge_index = data.edge_index[:, data.max_attention.squeeze() > attention_cut]   

    return dataset

def build_dataset(
    datatype_split=[100,10,10], 
    num_tracks=100, 
    num_layers=10, 
    detector_width=0.5, 
    ptcut=1, 
    cut_policy=1,
    **kwargs):

    dataset = [generate_toy_dataset(num_events, num_tracks, num_layers, detector_width, ptcut, cut_policy) for num_events in datatype_split]

    return dataset


def ignore_warning(warning: Type[Warning]):
    """
    Ignore a given warning occurring during method execution.
    Args:
        warning (Warning): warning type to ignore.
    Returns:
        the inner function
    """

    def inner(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category= warning)
                return func(*args, **kwargs)

        return wrapper

    return inner

def apply_geometric_cut(fully_connected_graph, node_feature, num_layers, detector_width, cut_policy = 1):

    if cut_policy >= 1:

        del_x = (node_feature[fully_connected_graph[1], 0] - node_feature[fully_connected_graph[0], 0])
        fully_connected_graph = fully_connected_graph[:, (del_x <= 2*detector_width/num_layers) & (del_x > 0)]
   
    fully_connected_graph = fully_connected_graph[:, node_feature[fully_connected_graph[0], 2] != node_feature[fully_connected_graph[1], 2]]

    return fully_connected_graph

def define_truth_graph(node_feature, ptcut):

    connections = (node_feature[:-1, 2] == node_feature[1:,2])
    idxs = np.arange(len(node_feature))

    truth_graph = np.vstack([idxs[:-1][connections], idxs[1:][connections]])
    signal_truth_graph = truth_graph[:, (node_feature[:, 3][truth_graph] > ptcut).all(0)]

    return truth_graph, signal_truth_graph

def construct_training_graph(node_feature, num_layers, detector_width, cut_policy):

    idxs = np.arange(len(node_feature))
    fully_connected_graph = np.vstack([np.resize(idxs, (len(idxs),len(idxs))).flatten(), np.resize(idxs, (len(idxs),len(idxs))).T.flatten()])
    fully_connected_graph = fully_connected_graph[:, np.random.choice(fully_connected_graph.shape[1], size = min(1000, len(node_feature))*len(node_feature), replace = False)]

    fully_connected_graph = apply_geometric_cut(fully_connected_graph, node_feature, num_layers, detector_width, cut_policy)   

    return fully_connected_graph

def generate_single_track(i, min_r, max_r, num_layers, detector_width):

    r = np.random.uniform(min_r, max_r)
    theta = np.random.uniform(0, np.pi)
    sign = np.random.choice([-1, 1])

    x = np.linspace(0.05, detector_width + 0.05, num = num_layers)
    y = sign*(np.sqrt(r**2 - (x - r*np.cos(theta))**2) - r*np.sin(theta))
    pid = np.array(len(x)*[i+1], dtype = np.int64)
    pt = 1000 * np.array(len(x)*[r])
    
    mask = (y == y)
    x, y, pid, pt = x[mask], y[mask], pid[mask], pt[mask]

    return np.vstack([x, y, pid, pt]).T

@ignore_warning(RuntimeWarning)
def generate_toy_event(num_tracks, num_layers, detector_width, ptcut, cut_policy):

    # 50% chance of signal event
    signal_event = np.random.choice([True, False])

    tracks = []
    i=0

    if signal_event:
        while i < num_tracks:
            track = generate_single_track(i, 0.0, 0.4, num_layers, detector_width)
            if len(track) and len(track.T[0]) > 5:
                    tracks.append(track)
                    i+=1
        while True:
            high_pt_track = generate_single_track(i, 0.8, 1.0, num_layers, detector_width)
            if len(high_pt_track) and len(high_pt_track.T[0]) > 5:
                tracks.append(high_pt_track)
                break
    else:
        while i < num_tracks + 1:
            track = generate_single_track(i, 0.0, 0.4, num_layers, detector_width)
            if len(track) and len(track.T[0]) > 5:
                    tracks.append(track)
                    i+=1


    # Stack together track features
    node_feature = np.concatenate(tracks, axis = 0)

    # Define truth and training graphs
    truth_graph, signal_true_graph = define_truth_graph(node_feature, ptcut) 
    graph = construct_training_graph(node_feature, num_layers, detector_width, cut_policy)
    graph = np.concatenate([graph, signal_true_graph], axis = 1)
    # Shuffle the graph
    graph = graph[:, np.random.choice(graph.shape[1], size = graph.shape[1], replace = False)]
    
    graph, y = graph_intersection(graph, signal_true_graph)
    node_feature = torch.from_numpy(node_feature).float()
    
    y_pid = (node_feature[:,2][graph[0]] == node_feature[:,2][graph[1]])
    pid_signal = (node_feature[:,2][graph[0]] == node_feature[:,2][graph[1]]) & (node_feature[:,3][graph]).all(0)
    
    event = Data(x=node_feature[:,0:2],
                 edge_index = graph,
                 modulewise_true_edges = torch.tensor(truth_graph).T,
                 signal_true_edges = torch.tensor(signal_true_graph).T,
                 y=y,
                 pt = node_feature[:,3],
                 pid = node_feature[:,2].long(),
                 y_pid = y_pid,
                 pid_signal = pid_signal,
                 y_trigger = torch.tensor(signal_event).bool()
                )
    
    return event

def generate_toy_dataset(num_events, num_tracks, num_layers, detector_width, ptcut, cut_policy):
    
    dataset = []
    
    for i in tqdm(range(num_events)):
        dataset.append(generate_toy_event(num_tracks, num_layers, detector_width, ptcut, cut_policy))
    return dataset


def graph_intersection(
    pred_graph, truth_graph
):

    array_size = max(pred_graph.max().item(), truth_graph.max().item()) + 1

    if torch.is_tensor(pred_graph):
        l1 = pred_graph.cpu().numpy()
    else:
        l1 = pred_graph
    if torch.is_tensor(truth_graph):
        l2 = truth_graph.cpu().numpy()
    else:
        l2 = truth_graph
    e_1 = sp.sparse.coo_matrix(
        (np.ones(l1.shape[1]), l1), shape=(array_size, array_size)
    ).tocsr()
    e_2 = sp.sparse.coo_matrix(
        (np.ones(l2.shape[1]), l2), shape=(array_size, array_size)
    ).tocsr()
    del l1

    e_intersection = e_1.multiply(e_2) - ((e_1 - e_2) > 0)
    del e_1
    del e_2

    e_intersection = e_intersection.tocoo()
    new_pred_graph = torch.from_numpy(
        np.vstack([e_intersection.row, e_intersection.col])
    ).long()  # .to(device)
    y = torch.from_numpy(e_intersection.data > 0)  # .to(device)
    del e_intersection
    
    return new_pred_graph, y