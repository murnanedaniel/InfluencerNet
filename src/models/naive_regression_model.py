import contextlib

# System imports
import sys

import torch

# Local Imports
from .object_condensation_base import ObjectCondensationBase
from torch_geometric.data import Batch
from torch_geometric.nn import aggr
from torch_geometric.utils import to_dense_batch

from ..losses.influencer_loss import influencer_loss

# from ..losses.regression_loss import regression_loss

device = "cuda" if torch.cuda.is_available() else "cpu"
sqrt_eps = 1e-12


class NaiveRegressionModel(ObjectCondensationBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """
        self.save_hyperparameters(hparams)
        self.median_aggr = aggr.MedianAggregation()
        self.loss_func = self.get_loss_func(hparams.get("loss_func", "mse"))

    def get_loss_func(self, loss_func):
        loss_funcs = {
            "mse": torch.nn.functional.mse_loss,
            "l1": torch.nn.functional.l1_loss,
            "smooth_l1": torch.nn.functional.smooth_l1_loss,
        }
        return loss_funcs[loss_func]

    def forward(self, x, **kwargs):
        return self.network(x, **kwargs)

    def get_regression_truth(self, batch):
        """
        1. Pull out all the regression targets from the batch as keys
        2. Concatenate them into a single tensor
        3. Get the cluster_ids of each node (row entry)
        """
        regression_truth = torch.cat(
            [batch[key] for key in self.hparams["regression_targets"]]
        )
        if len(regression_truth.shape) == 1:
            regression_truth = regression_truth.unsqueeze(-1)
        regression_truth = self.median_aggr(regression_truth, batch.batched_cluster_id)

        return regression_truth

    def training_step(self, batch, batch_idx):
        """
        The naive regression training step.
        1. Create a unique index for each cluster in the batch, from (batch_index, cluster_ID)
        2. Run all nodes through the regression transformer network
        3.
        """

        # Create unique batched cluster IDs
        batch.batched_cluster_id = batch.ptr[batch.batch] + batch.cluster_id
        _, batch.batched_cluster_id = torch.unique(
            batch.batched_cluster_id, return_inverse=True
        )

        # Get the follower and influencer embeddings
        input_data = self.get_input_data(batch)

        regression_truth = self.get_regression_truth(batch)
        regression_predictions = self(
            input_data,
            batch=batch.batched_cluster_id,
            edge_index=batch.get("edge_index"),
        )

        # Calculate the loss as MSE
        loss = self.loss_func(regression_predictions, regression_truth)
        mae = torch.nn.functional.l1_loss(regression_predictions, regression_truth)
        output_mean = regression_predictions.mean()
        output_std = regression_predictions.std()

        # Log
        log_dict = {
            "train_loss": loss,
            "train_mae": mae,
            "train_output_mean": output_mean,
            "train_output_std": output_std,
        }
        self.log_dict(log_dict, on_epoch=True, on_step=False)

        return loss

    def shared_evaluation(self, batch, batch_idx):
        # Create unique batched cluster IDs
        batch.batched_cluster_id = batch.ptr[batch.batch] + batch.cluster_id
        _, batch.batched_cluster_id = torch.unique(
            batch.batched_cluster_id, return_inverse=True
        )

        # Get the follower and influencer embeddings
        input_data = self.get_input_data(batch)

        regression_truth = self.get_regression_truth(batch)
        regression_predictions = self(
            input_data,
            batch=batch.batched_cluster_id,
            edge_index=batch.get("edge_index"),
        )

        # Calculate the loss as MSE
        loss = self.loss_func(regression_predictions, regression_truth)
        output_mean = regression_predictions.mean()

        # Log the loss, learning rate and absolute error
        lr = self.optimizers().param_groups[0]["lr"]
        abs_error = torch.nn.functional.l1_loss(
            regression_predictions, regression_truth
        )
        rel_error = (
            torch.abs(regression_predictions - regression_truth)
            / torch.abs(regression_truth)
        ).mean()
        log_dict = {
            "val_loss": loss,
            "lr": lr,
            "abs_error": abs_error,
            "rel_error": rel_error,
            "val_output_mean": output_mean,
        }
        self.log_dict(log_dict, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        return self.shared_evaluation(batch, batch_idx)
