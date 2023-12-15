import torch
import pandas as pd
import numpy as np
from torchmetrics import Accuracy, Precision, Recall, MetricCollection

from .object_condensation_base import ObjectCondensationBase
from torch_geometric.nn import aggr

device = "cuda" if torch.cuda.is_available() else "cpu"
sqrt_eps = 1e-12


def convert_targets_to_classes(targets, batch, num_bins, bin_limits):
    targets = targets.cpu().numpy().squeeze(-1)
    targets = np.clip(targets, bin_limits[0], bin_limits[1])
    bin_edges = np.linspace(bin_limits[0], bin_limits[1], num_bins + 1)
    target_classes = pd.cut(targets, bins=bin_edges, labels=False, include_lowest=True)
    target_classes = torch.from_numpy(target_classes).to(batch.pt_truth.device)
    return target_classes


class NaiveBinnedRegressionModel(ObjectCondensationBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.save_hyperparameters(hparams)
        self.median_aggr = aggr.MedianAggregation()
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.metrics = MetricCollection(
            {
                "Accuracy": Accuracy(
                    task="multiclass", num_classes=hparams["num_bins"], average="macro"
                ),
                "Precision": Precision(
                    task="multiclass", num_classes=hparams["num_bins"], average="macro"
                ),
                "Recall": Recall(
                    task="multiclass", num_classes=hparams["num_bins"], average="macro"
                ),
            }
        )

    def forward(self, x, **kwargs):
        logits = self.network(x[:, : self.hparams.get("spatial_channels")], **kwargs)
        return logits

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

    def get_classification_truth(self, regression_truth, batch):
        classification_truth = convert_targets_to_classes(
            regression_truth,
            batch,
            self.hparams["num_bins"],
            self.hparams["bin_limits"],
        )
        return classification_truth

    def training_step(self, batch, batch_idx):
        batch.batched_cluster_id = batch.ptr[batch.batch] + batch.cluster_id
        _, batch.batched_cluster_id = torch.unique(
            batch.batched_cluster_id, return_inverse=True
        )
        input_data = self.get_input_data(batch)
        regression_truth = self.get_regression_truth(batch)
        classification_truth = self.get_classification_truth(regression_truth, batch)
        classification_predictions = self(
            input_data,
            batch=batch.batched_cluster_id,
            edge_index=batch.get("edge_index"),
        )
        loss = self.loss_func(classification_predictions, classification_truth)
        self.log_dict({"train_loss": loss}, on_step=False, on_epoch=True)
        return loss

    def shared_evaluation(self, batch, batch_idx):
        batch.batched_cluster_id = batch.ptr[batch.batch] + batch.cluster_id
        _, batch.batched_cluster_id = torch.unique(
            batch.batched_cluster_id, return_inverse=True
        )
        input_data = self.get_input_data(batch)
        regression_truth = self.get_regression_truth(batch)
        classification_truth = self.get_classification_truth(regression_truth, batch)
        classification_predictions = self(
            input_data,
            batch=batch.batched_cluster_id,
            edge_index=batch.get("edge_index"),
        )
        loss = self.loss_func(classification_predictions, classification_truth)
        lr = self.optimizers().param_groups[0]["lr"]
        self.metrics.update(classification_predictions, classification_truth)
        self.log_dict({"val_loss": loss, "lr": lr}, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.shared_evaluation(batch, batch_idx)

    def on_validation_epoch_end(self):
        self.log_dict(self.metrics.compute())
        self.metrics.reset()
