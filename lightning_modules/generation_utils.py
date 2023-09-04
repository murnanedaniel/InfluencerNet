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

warnings.filterwarnings("ignore")
sys.path.append("../../..")
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_dataset(input_dir = "/global/cfs/cdirs/m3443/data/TrackLRP/toy_dataset_v1", datatype_split=None, **kwargs):

    if datatype_split is None:
        datatype_split = [500, 100, 10]
    subdirs = ["train", "val", "test"]

    return [load_toy_dataset(os.path.join(input_dir, subdir), num_events, **kwargs) for num_events, subdir in zip(datatype_split, subdirs)]

def load_toy_dataset(input_dir, num_events, **kwargs):

    # List num_events of files in input_dir
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    random.shuffle(files)
    files = files[:num_events]

    return [ torch.load(f) for f in files ]

def build_dataset(datatype_split=None, num_tracks=100, track_dis_width=10, num_layers=10, min_r=0.1, max_r=0.5, detector_width=0.5, ptcut=1, nhits=1, **kwargs):

    if datatype_split is None:
        datatype_split = [100,10,10]

    return [generate_toy_dataset(num_events, num_tracks, track_dis_width, num_layers, min_r, max_r, detector_width, ptcut, nhits) for num_events in datatype_split]

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

def generate_single_track(i, min_r, max_r, num_layers, detector_width):

    r = np.random.uniform(min_r, max_r)
    theta = np.random.uniform(0, np.pi)
    sign = np.random.choice([-1, 1])

    x = np.linspace(0.05, detector_width + 0.05, num = num_layers)
    y = sign*(np.sqrt(r**2 - (x - r*np.cos(theta))**2) - r*np.sin(theta))
    pid = np.array(len(x)*[np.random.randint(0, 2**32)], dtype = np.int64)
    pt = 1000 * np.array(len(x)*[r])
    
    mask = (y == y)
    x, y, pid, pt = x[mask], y[mask], pid[mask], pt[mask]

    return np.vstack([x, y, pid, pt]).T

def define_truth_graph(node_feature, ptcut):
    """
    There are four types of truth, combinations of:
    1. Sequential or all-same-PID
    2. Signal or signal+background
    """

    # Calculate sequential truth graph
    connections = (node_feature[:-1, 2] == node_feature[1:,2])
    idxs = np.arange(len(node_feature))

    seq_truth_graph = torch.from_numpy(np.vstack([idxs[:-1][connections], idxs[1:][connections]]))
    seq_signal_truth_graph = seq_truth_graph[:, (node_feature[:, 3][seq_truth_graph] > ptcut).all(0)]
    
    # Calculate all-same-PID truth graph
    all_edge_combinations = torch.cartesian_prod(torch.arange(len(node_feature)), torch.arange(len(node_feature))).T
    all_truth_graph = all_edge_combinations[:, node_feature[all_edge_combinations[0], 2] == node_feature[all_edge_combinations[1], 2]]
    
    # Check that all-same-PID truth graph is symmetric
    increasing = all_truth_graph[0] < all_truth_graph[1]
    equal = all_truth_graph[0] == all_truth_graph[1]
    decreasing = all_truth_graph[0] > all_truth_graph[1]
    assert increasing.sum() == decreasing.sum(), "Truth graph is not symmetric"

    # Check that all self-edges are included
    assert (all_truth_graph[0] == all_truth_graph[1]).sum() == len(node_feature), "Truth graph is missing self-edges"

    # Calculate signal truth graph
    sig_truth_graph = all_truth_graph[:, (node_feature[:, 3][all_truth_graph] > ptcut).all(0)]

    return seq_truth_graph, seq_signal_truth_graph, all_truth_graph, sig_truth_graph


def apply_nhits_min(event, nhits):

    _, inverse, counts = event.pid.unique(return_inverse = True, return_counts = True)
    event.nhits = counts[inverse]
    event.pid[(event.nhits < nhits)] = 0

    # event.seq_signal_edges = event.seq_signal_edges[:, (event.nhits[event.seq_signal_edges] > nhits).all(0)]
    event.edge_index = event.edge_index[:, (event.nhits[event.edge_index] > nhits).all(0)]

    return event

@ignore_warning(RuntimeWarning)
def generate_toy_event(num_tracks, track_dis_width, num_layers, min_r, max_r, detector_width, ptcut, nhits):
    # pT is defined as 1000r
    
    tracks = []
    num_tracks = random.randint(num_tracks-track_dis_width, num_tracks+track_dis_width)

    # Generate all the tracks
    for i in range(num_tracks):
        track = generate_single_track(i, min_r, max_r, num_layers, detector_width)
        tracks.append(track)

    # Stack together track features
    node_feature = np.concatenate(tracks, axis = 0)
    
    # Define truth and training graphs
    _, _, all_truth_graph, _ = define_truth_graph(node_feature, ptcut) 
    node_feature = torch.from_numpy(node_feature).float()
        
    event = Data(x=node_feature[:,0:2],
                #  seq_signal_edges = seq_signal_truth_graph,
                 edge_index = all_truth_graph,
                 pt = node_feature[:,3],
                 pid = node_feature[:,2].long(),
                )

    event = apply_nhits_min(event, nhits)    
    
    return event

def generate_toy_dataset(num_events, num_tracks, track_dis_width, num_layers, min_r, max_r, detector_width, ptcut, nhits):
    dataset = []
    for i in range(num_events):
        try:
            event = generate_toy_event(num_tracks, track_dis_width, num_layers, min_r, max_r, detector_width, ptcut, nhits)
            if event.pid.max() > 0 and len(event.x) > 2*num_tracks:
                dataset.append(event)
        except Exception as e:
            print("Error in event: ", i, "", e)
    return dataset