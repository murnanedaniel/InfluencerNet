import os, sys
import logging
import random
from pathlib import Path

import torch.nn as nn
import torch
import pandas as pd
import numpy as np
try:
    import cupy as cp
except:
    pass
from torch_scatter import scatter

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
    query, database, indices=None, r_max=1.0, k_max=10, return_indices=False, backend="PYG", self_loop=False, batch_index=None
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
        if batch_index.max() == 0:
            batch_index = None
        edge_list = radius(database, query, r=r_max, max_num_neighbors=k_max, batch_x=batch_index, batch_y=batch_index)

    # Reset indices subset to correct global index
    if indices is not None:
        edge_list[0] = indices[edge_list[0]]

    # Remove self-loops
    if not self_loop:
        edge_list = edge_list[:, edge_list[0] != edge_list[1]]

    return (edge_list, dists, idxs, ind) if return_indices else edge_list

def load_datafiles_in_dir(input_dir, data_name = None, data_num = None):

    if data_name is not None:
        input_dir = os.path.join(input_dir, data_name)

    data_files = [str(path) for path in Path(input_dir).rglob("*.pyg")]

    # This is new: Sort data files to get handle on train/val/test contamination
    data_files.sort()

    data_files = data_files[:data_num]

    assert data_files, f"No data files found in {input_dir}"
    if data_num is not None:
        assert len(data_files) == data_num, f"Number of data files found ({len(data_files)}) is less than the number requested ({data_num})"

    return data_files

def handle_hard_node_cuts(event, hard_cuts_config):
    """
    Given set of cut config, remove nodes that do not pass the cuts.
    Remap the track_edges to the new node list.
    """
    node_like_feature = [event[feature] for feature in event.keys if event.is_node_attr(feature)][0]
    node_mask = torch.ones_like(node_like_feature, dtype=torch.bool)

    # TODO: Refactor this to simply trim the true tracks and check which nodes are in the true tracks
    for condition_key, condition_val in hard_cuts_config.items():
        assert condition_key in event.keys, f"Condition key {condition_key} not found in event keys {event.keys}"
        condition_lambda = get_condition_lambda(condition_key, condition_val)
        value_mask = condition_lambda(event)
        node_val_mask = map_tensor_handler(value_mask, output_type="node-like", track_edges=event.track_edges, num_nodes=node_like_feature.shape[0], num_track_edges=event.track_edges.shape[1])
        node_mask = node_mask * node_val_mask

    logging.info(f"Masking the following number of nodes with the HARD CUT: {node_mask.sum()} / {node_mask.shape[0]}")
    
    # TODO: Refactor the below to use the remap_from_mask function
    num_nodes = event.num_nodes
    for feature in event.keys:
        if isinstance(event[feature], torch.Tensor) and event[feature].shape and event[feature].shape[0] == num_nodes:
            event[feature] = event[feature][node_mask]

    num_tracks = event.track_edges.shape[1]
    track_mask = node_mask[event.track_edges].all(0)
    node_lookup = torch.cumsum(node_mask, dim=0) - 1
    for feature in event.keys:
        if isinstance(event[feature], torch.Tensor) and event[feature].shape and event[feature].shape[-1] == num_tracks:
            event[feature] = event[feature][..., track_mask]

    event.track_edges = node_lookup[event.track_edges]
    event.num_nodes = node_mask.sum()

def get_condition_lambda(condition_key, condition_val):

    condition_dict = {
        "is": lambda event: event[condition_key] == condition_val,
        "is_not": lambda event: event[condition_key] != condition_val,
        "in": lambda event: torch.isin(event[condition_key], torch.tensor(condition_val[1], device=event[condition_key].device)),
        "not_in": lambda event: ~torch.isin(event[condition_key], torch.tensor(condition_val[1], device=event[condition_key].device)),
        "within": lambda event: (condition_val[0] <= event[condition_key].float()) & (event[condition_key].float() <= condition_val[1]),
        "not_within": lambda event: not ((condition_val[0] <= event[condition_key].float()) & (event[condition_key].float() <= condition_val[1])),
    }

    if isinstance(condition_val, bool):
        return lambda event: event[condition_key] == condition_val
    elif isinstance(condition_val, list) and not isinstance(condition_val[0], str):
        return lambda event: (condition_val[0] <= event[condition_key].float()) & (event[condition_key].float() <= condition_val[1])
    elif isinstance(condition_val, list):
        return condition_dict[condition_val[0]]
    else:
        raise ValueError(f"Condition {condition_val} not recognised")

def map_tensor_handler(input_tensor: torch.Tensor, 
                       output_type: str, 
                       input_type: str = None, 
                       truth_map: torch.Tensor = None, 
                       edge_index: torch.Tensor = None,
                       track_edges: torch.Tensor = None,
                       num_nodes: int = None, 
                       num_edges: int = None, 
                       num_track_edges: int = None,
                       aggr: str = None):
    """
    A general function to handle arbitrary maps of one tensor type to another. Types are "node-like", "edge-like" and "track-like".
    - Node-like: The input tensor is of the same size as the number of nodes in the graph
    - Edge-like: The input tensor is of the same size as the number of edges in the graph, that is, the *constructed* graph
    - Track-like: The input tensor is of the same size as the number of true track edges in the event, that is, the *truth* graph

    To visualize:
                    (n)
                     ^
                    / \ 
      edge_to_node /   \ track_to_node
                  /     \
                 /       \
                /         \
               /           \
              /             \
node_to_edge /               \ node_to_track
            /                 \
           |                   | 
           v     edge_to_track v
          (e) <-------------> (t)
            track_to_edge

    Args:
        input_tensor (torch.Tensor): The input tensor to be mapped
        output_type (str): The type of the output tensor. One of "node-like", "edge-like" or "track-like"
        input_type (str, optional): The type of the input tensor. One of "node-like", "edge-like" or "track-like". Defaults to None, and will try to infer the type from the input tensor, if num_nodes and/or num_edges are provided.
        truth_map (torch.Tensor, optional): The truth map tensor. Defaults to None. Used for mappings to/from track-like tensors.
        num_nodes (int, optional): The number of nodes in the graph. Defaults to None. Used for inferring the input type.
        num_edges (int, optional): The number of edges in the graph. Defaults to None. Used for inferring the input type.
        num_track_edges (int, optional): The number of track edges in the graph. Defaults to None. Used for inferring the input type.
    """

    # Refactor the above switch case into a dictionary
    mapping_dict = {
        ("node-like", "edge-like"): lambda input_tensor, truth_map, edge_index, track_index, num_nodes, num_edges, num_track_edges, aggr: map_nodes_to_edges(input_tensor, edge_index, aggr),
        ("edge-like", "node-like"): lambda input_tensor, truth_map, edge_index, track_index, num_nodes, num_edges, num_track_edges, aggr: map_edges_to_nodes(input_tensor, edge_index, aggr, num_nodes),
        ("node-like", "track-like"): lambda input_tensor, truth_map, edge_index, track_index, num_nodes, num_edges, num_track_edges, aggr: map_nodes_to_tracks(input_tensor, track_edges, aggr),
        ("track-like", "node-like"): lambda input_tensor, truth_map, edge_index, track_index, num_nodes, num_edges, num_track_edges, aggr: map_tracks_to_nodes(input_tensor, track_edges, aggr, num_nodes),
        ("edge-like", "track-like"): lambda input_tensor, truth_map, edge_index, track_index, num_nodes, num_edges, num_track_edges, aggr: map_edges_to_tracks(input_tensor, truth_map),
        ("track-like", "edge-like"): lambda input_tensor, truth_map, edge_index, track_index, num_nodes, num_edges, num_track_edges, aggr: map_tracks_to_edges(input_tensor, truth_map, num_edges),
    }

    if num_track_edges is None and truth_map is not None:
        num_track_edges = truth_map.shape[0]
    if num_track_edges is None and track_edges is not None:
        num_track_edges = track_edges.shape[1]
    if num_edges is None and edge_index is not None:
        num_edges = edge_index.shape[1]
    if input_type is None:
        input_type, input_tensor = infer_input_type(input_tensor, num_nodes, num_edges, num_track_edges)

    if input_type == output_type:
        return input_tensor
    elif (input_type, output_type) in mapping_dict:
        return mapping_dict[(input_type, output_type)](input_tensor, truth_map, edge_index, track_edges, num_nodes, num_edges, num_track_edges, aggr)
    else:
        raise ValueError(f"Mapping from {input_type} to {output_type} not supported")
    
def infer_input_type(input_tensor: torch.Tensor, num_nodes: int = None, num_edges: int = None, num_track_edges: int = None):
    """
    Tries to infer the input type from the input tensor and the number of nodes, edges and track-edges in the graph.
    If the input tensor cannot be matched to any of the provided types, it is assumed to be node-like.
    """

    if num_nodes is not None and input_tensor.shape[0] == num_nodes:
        return "node-like", input_tensor
    elif num_edges is not None and num_edges in input_tensor.shape:
        return "edge-like", input_tensor
    elif num_track_edges is not None and num_track_edges in input_tensor.shape:
        return "track-like", input_tensor
    elif num_track_edges is not None and num_track_edges//2 in input_tensor.shape:
        return "track-like", torch.cat([input_tensor, input_tensor], dim=0)
    else:
        return "node-like", input_tensor


def map_nodes_to_edges(nodelike_input: torch.Tensor, edge_index: torch.Tensor, aggr: str = None):
    """
    Map a node-like tensor to an edge-like tensor. If the aggregation is None, this is simply done by sending node values to the edges, thus returning a tensor of shape (2, num_edges).
    If the aggregation is not None, the node values are aggregated to the edges, and the resulting tensor is of shape (num_edges,).
    """

    if aggr is None:
        return nodelike_input[edge_index]
    
    edgelike_tensor = nodelike_input[edge_index]
    torch_aggr = getattr(torch, aggr)
    return torch_aggr(edgelike_tensor, dim=0)
    
def map_edges_to_nodes(edgelike_input: torch.Tensor, edge_index: torch.Tensor, aggr: str = None, num_nodes: int = None):
    """
    Map an edge-like tensor to a node-like tensor. If the aggregation is None, this is simply done by sending edge values to the nodes, thus returning a tensor of shape (num_nodes,).
    If the aggregation is not None, the edge values are aggregated to the nodes at the destination node (edge_index[1]), and the resulting tensor is of shape (num_nodes,).
    """

    if num_nodes is None:
        num_nodes = int(edge_index.max().item() + 1)

    if aggr is None:
        nodelike_output = torch.zeros(num_nodes, dtype=edgelike_input.dtype, device=edgelike_input.device)
        nodelike_output[edge_index] = edgelike_input
        return nodelike_output
    
    return scatter(edgelike_input, edge_index[1], dim=0, dim_size=num_nodes, reduce=aggr)

def map_nodes_to_tracks(nodelike_input: torch.Tensor, track_edges: torch.Tensor, aggr: str = None):
    """
    Map a node-like tensor to a track-like tensor. If the aggregation is None, this is simply done by sending node values to the tracks, thus returning a tensor of shape (2, num_track_edges).
    If the aggregation is not None, the node values are aggregated to the tracks, and the resulting tensor is of shape (num_track_edges,).
    """
    
    if aggr is None:
        return nodelike_input[track_edges]
    
    tracklike_tensor = nodelike_input[track_edges]
    torch_aggr = getattr(torch, aggr)
    return torch_aggr(tracklike_tensor, dim=0)

def map_tracks_to_nodes(tracklike_input: torch.Tensor, track_edges: torch.Tensor, aggr: str = None, num_nodes: int = None):
    """
    Map a track-like tensor to a node-like tensor. If the aggregation is None, this is simply done by sending track values to the nodes, thus returning a tensor of shape (num_nodes,).
    If the aggregation is not None, the track values are aggregated to the nodes at the destination node (track_edges[1]), and the resulting tensor is of shape (num_nodes,).
    """

    if num_nodes is None:
        num_nodes = int(track_edges.max().item() + 1)

    if aggr is None:
        nodelike_output = torch.zeros(num_nodes, dtype=tracklike_input.dtype, device=tracklike_input.device)
        nodelike_output[track_edges] = tracklike_input
        return nodelike_output
    
    return scatter(tracklike_input.repeat(2), torch.cat([track_edges[0], track_edges[1]]), dim=0, dim_size=num_nodes, reduce=aggr)
    
def map_tracks_to_edges(tracklike_input: torch.Tensor, truth_map: torch.Tensor, num_edges: int = None):
    """
    Map an track-like tensor to a edge-like tensor. This is done by sending the track value through the truth map, where the truth map is >= 0. Note that where truth_map == -1,
    the true edge has not been constructed in the edge_index. In that case, the value is set to NaN.
    """

    if num_edges is None:
        num_edges = int(truth_map.max().item() + 1)
    edgelike_output = torch.zeros(num_edges, dtype=tracklike_input.dtype, device=tracklike_input.device)
    edgelike_output[truth_map[truth_map >= 0]] = tracklike_input[truth_map >= 0]
    try:
        edgelike_output[truth_map[truth_map == -1]] = float("nan")
    except Exception:
        print("Warning: Could not set NaN values in edgelike_output. This is probably because the truth_map is not a long tensor.", Exception)
    return edgelike_output

def map_edges_to_tracks(edgelike_input: torch.Tensor, truth_map: torch.Tensor):
    """
    TODO: Implement this. I don't think it is a meaningful operation, but it is needed for completeness.
    """
    raise NotImplementedError("This is not a meaningful operation, but it is needed for completeness")

def remap_from_mask(event, edge_mask):
    """ 
    Takes a mask applied to the edge_index tensor, and remaps the truth_map tensor indices to match.
    """

    truth_map_to_edges = torch.ones(edge_mask.shape[0], dtype=torch.long) * -1
    truth_map_to_edges[event.truth_map[event.truth_map >= 0]] = torch.arange(event.truth_map.shape[0])[event.truth_map >= 0]
    truth_map_to_edges = truth_map_to_edges[edge_mask]

    new_map = torch.ones(event.truth_map.shape[0], dtype=torch.long) * -1
    new_map[truth_map_to_edges[truth_map_to_edges >= 0]] = torch.arange(truth_map_to_edges.shape[0])[truth_map_to_edges >= 0]
    event.truth_map = new_map.to(event.truth_map.device)


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

