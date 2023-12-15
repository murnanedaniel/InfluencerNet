"""The base classes for the embedding process.
"""


import contextlib

# System imports
import sys

import torch
import pandas as pd

# Local Imports
from .object_condensation_base import ObjectCondensationBase
from .utils import build_edges
from torch_geometric.data import Dataset, Batch
import copy

device = "cuda" if torch.cuda.is_available() else "cpu"
sqrt_eps = 1e-12


class NaiveModel(ObjectCondensationBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """
        self.save_hyperparameters(hparams)

    def get_training_edges(
        self,
        batch,
        embedding_query,
        embedding_database,
        hnm=False,
        rp=False,
        tp=False,
        radius=None,
        knn=None,
        batch_index=None,
        self_loop=None,
    ):
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
            training_edges = self.append_hnm_pairs(
                training_edges,
                embedding_query,
                embedding_database,
                radius=radius,
                knn=knn,
                batch_index=batch_index,
                self_loop=self_loop,
            )

        # Append Random Pairs (rp) with KNN graph
        if rp:
            training_edges = self.append_random_pairs(
                training_edges,
                embedding_query,
                embedding_database,
                batch_index=batch_index,
            )

        # Append True Pairs (tp) with KNN graph
        if tp:
            training_edges = self.append_true_pairs(training_edges, batch)

        # Remove duplicates
        training_edges = training_edges.unique(dim=1)

        # Get the truth values for the edges
        training_truth = self.get_truth(training_edges, batch)

        return training_edges, training_truth

    def get_follower_follower_loss(self, follower_follower_edges, follower_follower_truth, follower_embed):
        hinge, d = self.get_hinge_distance(
            follower_embed, follower_embed, follower_follower_edges, follower_follower_truth
        )

        if (hinge == -1).any():
            negative_loss = torch.nn.functional.hinge_embedding_loss(
                d[hinge == -1],
                hinge[hinge == -1],
                # margin=self.hparams["margin"]**2, # SQREUCLIDEAN
                margin=self.hparams["margin"],  # EUCLIDEAN
                reduction="mean",
            )
        else:
            negative_loss = 0

        if (hinge == 1).any():
            positive_loss = torch.nn.functional.hinge_embedding_loss(
                d[hinge == 1] ** 2,
                hinge[hinge == 1],
                margin=self.hparams["margin"] ** 2,  # SQREUCLIDEAN
                # margin=self.hparams["margin"], # EUCLIDEAN
                reduction="mean",
            )
        else:
            positive_loss = 0

        return negative_loss + self.hparams["follower_follower_pos_ratio"] * positive_loss

    def training_step(self, batch, batch_idx):
        """
        The OC training step.
        1. Runs the model in no_grad mode to get the follower and influencer embeddings
        2. Builds hard negative, random pairs, and true edges for the follower-follower loss
        3. Calculate the "follower-follower" loss
        """

        # Get the follower and influencer embeddings
        input_data = self.get_input_data(batch)
        follower_embed = self(input_data, batch=batch.batch)
        follower_follower_edges, follower_follower_truth = self.get_training_edges(
            batch,
            follower_embed,
            follower_embed,
            hnm=True,
            rp=True,
            tp=True,
            batch_index=batch.batch,
        )

        # Calculate the total loss
        loss = self.get_follower_follower_loss(follower_follower_edges, follower_follower_truth, follower_embed)

        self.log_dict({"train_loss": loss}, on_epoch=True)

        if torch.isnan(loss):
            print("Loss is nan")
            sys.exit()

        return loss

    def get_condensation_edges(self, batch, follower_embed):
        # Build candidate edges as radius graph with all vertices as query and database
        candidate_edges = build_edges(
            follower_embed,
            follower_embed,
            r_max=self.hparams["radius"],
            k_max=100,
            batch_index=batch.batch,
            self_loop=True,
        )

        # Loop over vertices, and add edges to final list if vertex is not already in an edge
        condensation_edges = torch.empty([2, 0], dtype=torch.int64, device=self.device)

        for condensation_index in torch.randperm(follower_embed.shape[0]):
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
        """
        This method is used for shared evaluation of the model. It calculates the loss and various metrics for the model.
        It also logs the metrics and plots the embeddings.

        Args:
            batch (torch.Tensor): The batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            dict: A dictionary containing the loss and various metrics.
        """

        # Get the input data
        input_data = self.get_input_data(batch)

        # Start validation tracking
        self.start_validation_tracking()

        # Get the follower embeddings
        follower_embed = self(input_data, batch.batch)

        # Get the condensation edges and their truth values
        follower_influencer_edges, follower_influencer_truth = self.get_condensation_edges(
            batch, follower_embed
        )

        # End validation tracking
        self.end_validation_tracking()

        # Try to get the training edges and calculate the loss
        try:
            follower_follower_edges, follower_follower_truth = self.get_training_edges(
                batch,
                follower_embed,
                follower_embed,
                hnm=True,
                knn=500,
                self_loop=True,
                batch_index=batch.batch,
            )
            loss = self.get_follower_follower_loss(follower_follower_edges, follower_follower_truth, follower_embed)
        except Exception:
            # If an exception occurs, initialize empty edges and zero loss
            follower_follower_edges, follower_follower_truth = torch.empty(
                [2, 0], dtype=torch.int64, device=self.device
            ), torch.empty([0], dtype=torch.int64, device=self.device)
            loss = torch.tensor(0, dtype=torch.float32, device=self.device)

        # Get the current learning rate
        current_lr = self.optimizers().param_groups[0]["lr"]

        # Calculate various metrics
        cluster_eff, cluster_pur = self.get_cluster_metrics(
            batch, follower_follower_edges, follower_follower_truth
        )
        represent_eff, represent_pur, represent_dup = self.get_representative_metrics(
            batch, follower_influencer_edges, follower_influencer_truth
        )
        tracking_eff, tracking_pur, tracking_dup = self.get_tracking_metrics(
            batch, follower_influencer_edges, follower_influencer_truth
        )

        # Try to log the metrics
        with contextlib.suppress(Exception):
            self.log_dict(
                {
                    "val_loss": loss,
                    "represent_pur": represent_pur,
                    "represent_eff": represent_eff,
                    "represent_dup": represent_dup,
                    "cluster_pur": cluster_pur,
                    "cluster_eff": cluster_eff,
                    "tracking_fake_rate": 1 - tracking_pur,
                    "tracking_eff": tracking_eff,
                    "tracking_dup": tracking_dup,
                    "lr": current_lr,
                },
            )

        # If it's the first batch, print some metrics and log the embedding plot
        if batch_idx == 0:
            print(
                f"Rep eff: {represent_eff}, rep pur: {represent_pur}, rep dup: {represent_dup}"
            )
            print(f"Clu eff: {cluster_eff}, clu pur: {cluster_pur}")

            first_event = Batch.to_data_list(batch)[0]
            batch_mask = batch.batch == 0

            self.log_embedding_plot(
                batch,
                follower_embed[batch_mask],
                spatial2=follower_embed[batch_mask],
                ui_edges=follower_influencer_edges[
                    :, batch_mask[follower_influencer_edges].all(dim=0)
                ],
                uu_edges=follower_follower_edges[:, batch_mask[follower_follower_edges].all(dim=0)],
            )

        # Return the loss and metrics
        return {
            "val_loss": loss,
            "represent_pur": represent_pur,
            "represent_eff": represent_eff,
            "represent_dup": represent_dup,
            "follower_follower_edges": follower_follower_edges,
            "follower_follower_truth": follower_follower_truth,
            "follower_embed": follower_embed,
            "follower_influencer_edges": follower_influencer_edges,
            "follower_influencer_truth": follower_influencer_truth,
        }
