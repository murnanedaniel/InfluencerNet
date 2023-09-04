"""The base classes for the embedding process.
"""


import contextlib
# System imports
import sys

import torch
import pandas as pd

# Local Imports
from .influencer_base import InfluencerBase
from .utils import build_edges
from torch_geometric.data import Dataset, Batch
import copy

from .generation_utils import graph_intersection, generate_toy_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
sqrt_eps = 1e-12

class NaiveBase(InfluencerBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """
        self.save_hyperparameters(hparams)

    def get_training_edges(self, batch, embedding_query, embedding_database, hnm=False, rp=False, tp=False, radius=None, knn=None, batch_index=None):

        """
        Builds the edges for the training step. 
        1. Builds hard negative pairs if hnm is True
        2. Builds random pairs if rp is True
        3. Builds true pairs if tp is True
        """

        # Instantiate empty prediction edge list
        training_edges = torch.empty([2, 0], dtype=torch.int64, device=self.device)

        # Append Hard Negative Mining (hnm) with KNN graph
        if hnm:
            training_edges = self.append_hnm_pairs(training_edges, embedding_query, embedding_database, radius=radius, knn=knn, batch_index=batch_index)

        # Append Random Pairs (rp) with KNN graph
        if rp:
            training_edges = self.append_random_pairs(training_edges, embedding_query, embedding_database, batch_index=batch_index)

        # Append True Pairs (tp) with KNN graph
        if tp:
            training_edges = self.append_true_pairs(training_edges, batch)

        # Remove duplicates
        training_edges = training_edges.unique(dim=1)

        # Get the truth values for the edges
        training_truth = self.get_truth(training_edges, batch)

        return training_edges, training_truth

    def get_user_user_loss(self, user_user_edges, user_user_truth, user_embed):

        hinge, d = self.get_hinge_distance(user_embed, user_embed, user_user_edges, user_user_truth)

        if (hinge == -1).any():
            negative_loss = torch.nn.functional.hinge_embedding_loss(
                d[hinge == -1], 
                hinge[hinge == -1],
                # margin=self.hparams["margin"]**2, # SQREUCLIDEAN
                margin=self.hparams["margin"], # EUCLIDEAN                
                reduction="mean",
            )
        else:
            negative_loss = 0

        if (hinge == 1).any():
            positive_loss = torch.nn.functional.hinge_embedding_loss(
                d[hinge == 1]**2, 
                hinge[hinge == 1],
                margin=self.hparams["margin"]**2, # SQREUCLIDEAN
                # margin=self.hparams["margin"], # EUCLIDEAN
                reduction="mean",
            )
        else:
            positive_loss = 0

        return negative_loss + self.hparams["user_user_pos_ratio"] * positive_loss


    def training_step(self, batch, batch_idx):

        """
        The OC training step. 
        1. Runs the model in no_grad mode to get the user and influencer embeddings
        2. Builds hard negative, random pairs, and true edges for the user-user loss
        3. Calculate the "user-user" loss
        """

        # Get the user and influencer embeddings
        input_data = self.get_input_data(batch)
        user_embed = self(input_data, batch.batch)
        user_user_edges, user_user_truth = self.get_training_edges(batch, user_embed, user_embed, hnm=True, rp=True, tp=True, batch_index = batch.batch)

        # Calculate the total loss
        loss = self.get_user_user_loss(user_user_edges, user_user_truth, user_embed)

        self.log_dict({"train_loss": loss}, on_epoch=True)
        
        if torch.isnan(loss):
            print("Loss is nan")
            sys.exit()

        return loss

    def get_condensation_edges(self, batch, user_embed):

        # Build candidate edges as radius graph with beta vertices as database and all vertices as query
        candidate_edges = build_edges(user_embed, user_embed, r_max=self.hparams["radius"], k_max=100, batch_index = batch.batch)

        # Loop over vertices, and add edges to final list if vertex is not already in an edge
        condensation_edges = torch.empty([2, 0], dtype=torch.int64, device=self.device)

        for condensation_index in torch.randperm(user_embed.shape[0]):
            # If the vertex is already in an edge, skip it
            if condensation_index in condensation_edges:
                continue
            # Get the edges for the vertex
            vertex_edges = candidate_edges[:, candidate_edges[1] == condensation_index]
            # Add the edges to the final list
            condensation_edges = torch.cat([condensation_edges, vertex_edges], dim=1)

        # Get the truth values for the edges
        condensation_truth = self.get_truth(condensation_edges, batch)

        return condensation_edges, condensation_truth

    def shared_evaluation(self, batch, batch_idx):

        input_data = self.get_input_data(batch)
        self.start_validation_tracking()
        user_embed = self(input_data, batch.batch)
        
        # Get the condensation edges
        user_influencer_edges, user_influencer_truth = self.get_condensation_edges(batch, user_embed)
        self.end_validation_tracking()

        try:
            user_user_edges, user_user_truth = self.get_training_edges(batch, user_embed, user_embed, hnm=True, knn=500, batch_index=batch.batch)
            loss = self.get_user_user_loss(user_user_edges, user_user_truth, user_embed)
        except Exception:
            user_user_edges, user_user_truth = torch.empty([2, 0], dtype=torch.int64, device=self.device), torch.empty([0], dtype=torch.int64, device=self.device)
            loss = torch.tensor(0, dtype=torch.float32, device=self.device)

        current_lr = self.optimizers().param_groups[0]["lr"]        
        cluster_eff, cluster_pur = self.get_cluster_metrics(batch, user_user_edges, user_user_truth)
        represent_eff, represent_pur, represent_dup = self.get_representative_metrics(batch, user_influencer_edges, user_influencer_truth)
        tracking_eff, tracking_pur, tracking_dup = self.get_tracking_metrics(batch, user_influencer_edges, user_influencer_truth)

        with contextlib.suppress(Exception):
            self.log_dict(
                {
                    "val_loss": loss,
                    "represent_pur": represent_pur,
                    "represent_eff": represent_eff,
                    "represent_dup": represent_dup,
                    "cluster_pur": cluster_pur,
                    "cluster_eff": cluster_eff,
                    "tracking_fake_rate": 1-tracking_pur,
                    "tracking_eff": tracking_eff,
                    "tracking_dup": tracking_dup,
                    "lr": current_lr,
                },
            )
        if batch_idx == 0:
            print(f"Rep eff: {represent_eff}, rep pur: {represent_pur}, rep dup: {represent_dup}")
            print(f"Clu eff: {cluster_eff}, clu pur: {cluster_pur}")

            first_event = Batch.to_data_list(batch)[0]
            batch_mask = batch.batch == 0

            self.log_embedding_plot(batch, user_embed[batch_mask], spatial2=user_embed[batch_mask], ui_edges=user_influencer_edges[:, batch_mask[user_influencer_edges].all(dim=0)], uu_edges=user_user_edges[:, batch_mask[user_user_edges].all(dim=0)])

        return {"val_loss": loss, "represent_pur": represent_pur, "represent_eff": represent_eff, "represent_dup": represent_dup,
                "user_user_edges": user_user_edges, "user_user_truth": user_user_truth, "user_embed": user_embed,
                "user_influencer_edges": user_influencer_edges, "user_influencer_truth": user_influencer_truth}

    
