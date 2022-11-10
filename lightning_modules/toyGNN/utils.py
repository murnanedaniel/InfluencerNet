import os, sys
import logging
import random

import torch.nn as nn
import torch
import pandas as pd
import numpy as np
try:
    import cupy as cp
except:
    pass

from tqdm import tqdm
from torch_geometric.nn import radius

try:
    import frnn

    using_faiss = False
except ImportError:
    using_faiss = True

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    using_faiss = True

from torch_geometric.data import Dataset

# ---------------------------- Dataset Processing -------------------------


def load_dataset(
    input_subdir="",
    num_events=10,
    pt_background_cut=0,
    pt_signal_cut=0,
    noise=False,
    triplets=False,
    input_cut=None,
    **kwargs
):
    if input_subdir is not None:
        all_events = os.listdir(input_subdir)
        if "sorted_events" in kwargs.keys() and kwargs["sorted_events"]:
            all_events = sorted(all_events)
        else:
            random.shuffle(all_events)
        
        all_events = [os.path.join(input_subdir, event) for event in all_events]    
        print(f"Loading events from {input_subdir}")
        
        loaded_events = []
        for event in tqdm(all_events[:num_events]):
            loaded_events.append(torch.load(event, map_location=torch.device("cpu")))
        
        print("Events loaded!")
        
        loaded_events = process_data(
            loaded_events, pt_background_cut, pt_signal_cut, noise, triplets, input_cut
        )
        
        print("Events processed!")
        return loaded_events
    else:
        return None


def process_data(events, pt_background_cut, pt_signal_cut, noise, triplets, input_cut):
    # Handle event in batched form
    if type(events) is not list:
        events = [events]

    # NOTE: Cutting background by pT BY DEFINITION removes noise
    if pt_background_cut > 0 or not noise:
        for i, event in tqdm(enumerate(events)):

            if triplets:  # Keep all event data for posterity!
                event = convert_triplet_graph(event)

            else:
                event = background_cut_event(event, pt_background_cut, pt_signal_cut)                
                    
    for i, event in tqdm(enumerate(events)):
        
        # Ensure PID definition is correct
        event.y_pid = (event.pid[event.edge_index[0]] == event.pid[event.edge_index[1]]) & event.pid[event.edge_index[0]].bool()
        event.pid_signal = torch.isin(event.edge_index, event.signal_true_edges).all(0) & event.y_pid
        
        if (input_cut is not None) and "scores" in event.keys:
            score_mask = event.scores > input_cut
            for edge_attr in ["edge_index", "y", "y_pid", "pid_signal", "scores"]:
                event[edge_attr] = event[edge_attr][..., score_mask]            

    return events

def background_cut_event(event, pt_background_cut=0, pt_signal_cut=0):
    edge_mask = ((event.pt[event.edge_index] > pt_background_cut) & (event.pid[event.edge_index] == event.pid[event.edge_index]) & (event.pid[event.edge_index] != 0)).any(0)
    event.edge_index = event.edge_index[:, edge_mask]
    event.y = event.y[edge_mask]

    if "y_pid" in event.__dict__.keys():
        event.y_pid = event.y_pid[edge_mask]

    if "weights" in event.__dict__.keys():
        if event.weights.shape[0] == edge_mask.shape[0]:
            event.weights = event.weights[edge_mask]

    if (
        "signal_true_edges" in event.__dict__.keys()
        and event.signal_true_edges is not None
    ):
        signal_mask = (
            event.pt[event.signal_true_edges] > pt_signal_cut
        ).all(0)
        event.signal_true_edges = event.signal_true_edges[:, signal_mask]

    return event

def build_edges(
    query, database, indices=None, r_max=1.0, k_max=10, return_indices=False, backend="FRNN", self_loop=False
):

    if backend == "FRNN":
        try:
            dists, idxs, nn, grid = frnn.frnn_grid_points(
                points1=query.unsqueeze(0),
                points2=database.unsqueeze(0),
                lengths1=None,
                lengths2=None,
                K=k_max,
                r=r_max,
                grid=None,
                return_nn=False,
                return_sorted=True,
            )      

            idxs = idxs.squeeze().int()
            ind = torch.Tensor.repeat(
            torch.arange(idxs.shape[0], device=device), (idxs.shape[1], 1), 1
            ).T.int()
            positive_idxs = idxs >= 0
            edge_list = torch.stack([ind[positive_idxs], idxs[positive_idxs]]).long()
        except Exception:
            print("Radius error")
            return torch.zeros((2, 0), dtype=torch.long, device=device)

    elif backend == "PYG":
        edge_list = radius(database, query, r=r_max, max_num_neighbors=k_max)

    # Reset indices subset to correct global index
    if indices is not None:
        edge_list[0] = indices[edge_list[0]]

    # Remove self-loops
    if not self_loop:
        edge_list = edge_list[:, edge_list[0] != edge_list[1]]

    return (edge_list, dists, idxs, ind) if return_indices else edge_list

class LargeDataset(Dataset):
    def __init__(self, root, subdir, num_events, hparams, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.subdir = subdir
        self.hparams = hparams
        
        self.input_paths = os.listdir(os.path.join(root, subdir))
        if "sorted_events" in hparams.keys() and hparams["sorted_events"]:
            self.input_paths = sorted(self.input_paths)
        else:
            random.shuffle(self.input_paths)
        
        self.input_paths = [os.path.join(root, subdir, event) for event in self.input_paths][:num_events]
        
    def len(self):
        return len(self.input_paths)

    def get(self, idx):
        event = torch.load(self.input_paths[idx], map_location=torch.device("cpu"))
        
        # Process event with pt cuts
        if self.hparams["pt_background_cut"] > 0:
            event = background_cut_event(event, self.hparams["pt_background_cut"], self.hparams["pt_signal_cut"])

        # Ensure PID definition is correct
        event.y_pid = (event.pid[event.edge_index[0]] == event.pid[event.edge_index[1]]) & event.pid[event.edge_index[0]].bool()
        event.pid_signal = torch.isin(event.edge_index, event.signal_true_edges).all(0) & event.y_pid
        
        # if ("delta_eta" in self.hparams.keys()) and ((self.subdir == "train") or (self.subdir == "val" and self.hparams["n_graph_iters"] == 0)):
        if "delta_eta" in self.hparams.keys():
            eta_mask = hard_eta_edge_slice(self.hparams["delta_eta"], event)
            for edge_attr in ["edge_index", "y", "y_pid", "pid_signal", "scores"]:
                if edge_attr in event.keys:
                    event[edge_attr] = event[edge_attr][..., eta_mask]   
            
        if ("input_cut" in self.hparams.keys()) and (self.hparams["input_cut"] is not None) and "scores" in event.keys:
            score_mask = event.scores > self.hparams["input_cut"]
            for edge_attr in ["edge_index", "y", "y_pid", "pid_signal", "scores"]:
                if edge_attr in event.keys:
                    event[edge_attr] = event[edge_attr][..., eta_mask]
                
        return event
 

# ------------------------- Convenience Utilities ---------------------------


def make_mlp(
    input_size,
    sizes,
    hidden_activation="ReLU",
    output_activation=None,
    layer_norm=False,
    batch_norm=False,
):
    """Construct an MLP with specified fully-connected layers."""
    hidden_activation = getattr(nn, hidden_activation)
    if output_activation is not None:
        output_activation = getattr(nn, output_activation)
    layers = []
    n_layers = len(sizes)
    sizes = [input_size] + sizes
    # Hidden layers
    for i in range(n_layers - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[i + 1], elementwise_affine=False))
        if batch_norm:
            layers.append(nn.BatchNorm1d(sizes[i + 1], track_running_stats=False, affine=False))
        layers.append(hidden_activation())
    # Final layer
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if layer_norm and sizes[-1] > 1:
            layers.append(nn.LayerNorm(sizes[-1], elementwise_affine=False))
        if batch_norm:
            layers.append(nn.BatchNorm1d(sizes[-1], track_running_stats=False, affine=False))
        layers.append(output_activation())
    return nn.Sequential(*layers)

