import contextlib

# System imports
import sys

import torch

# Local Imports
from .influencer_model import InfluencerModel
from torch_geometric.data import Batch
from torch_geometric.nn import aggr

from ..losses.influencer_loss import influencer_loss

# from ..losses.regression_loss import regression_loss

device = "cuda" if torch.cuda.is_available() else "cpu"
sqrt_eps = 1e-12


class InfluencerRegressionModel(InfluencerModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """
        self.save_hyperparameters(hparams)

    def training_step(self, batch, batch_idx):
        """
        The Influencer training step.
        1. Runs the model in no_grad mode to get the follower and influencer embeddings
        2. Builds hard negative, random pairs, and true edges for the follower-follower loss
        3. Build hard negatives and true edges for the follower-influencer loss
        4. Build true edges for the influencer-influencer loss
        5. Compute the influencer loss
        6. Obtains influencer hits that have neighbors
        7. Gets the regression prediction for those hits
        8. Computes the regression loss
        """

        # Get the follower and influencer embeddings
        input_data = self.get_input_data(batch)
        with torch.no_grad():
            follower_embed, influencer_embed, regression_output = self(
                input_data, batch.batch
            )

        # Get the training edges for each loss function
        follower_follower_edges, follower_follower_truth = self.get_training_edges(
            batch,
            follower_embed,
            follower_embed,
            hnm=True,
            rp=True,
            tp=True,
            batch_index=batch.batch,
        )
        follower_influencer_edges, follower_influencer_truth = self.get_training_edges(
            batch,
            follower_embed,
            influencer_embed,
            hnm=True,
            tp=True,
            rp=True,
            batch_index=batch.batch,
        )
        (
            influencer_influencer_edges,
            influencer_influencer_truth,
        ) = self.get_training_edges(
            batch,
            influencer_embed,
            influencer_embed,
            hnm=True,
            rp=True,
            radius=self.hparams["influencer_margin"],
            batch_index=batch.batch,
        )

        # Get the hits of interest
        included_hits = torch.cat(
            [follower_follower_edges, follower_influencer_edges, influencer_influencer_edges], dim=1
        ).unique()
        (
            follower_embed[included_hits],
            influencer_embed[included_hits],
            regression_output[included_hits],
        ) = self(input_data[included_hits], batch.batch[included_hits])

        # Calculate each loss function

        infl_loss, sublosses = influencer_loss(
            follower_embed,
            influencer_embed,
            batch,
            follower_follower_edges,
            follower_follower_truth,
            follower_influencer_edges,
            follower_influencer_truth,
            influencer_influencer_edges,
            influencer_influencer_truth,
            follower_follower_weight=self.follower_follower_weight,
            follower_influencer_weight=self.follower_influencer_weight,
            influencer_influencer_weight=self.influencer_influencer_weight,
            follower_follower_pos_ratio=self.hparams["follower_follower_pos_ratio"],
            follower_influencer_neg_ratio=self.hparams["follower_influencer_neg_ratio"],
            follower_margin=self.hparams["margin"],
            influencer_margin=self.hparams["influencer_margin"],
            device=self.device,
            scatter_loss=self.hparams["scatter_loss"],
        )

        influential_hits = torch.unique(follower_influencer_edges[1])
        influencers_predictions = regression_output[influential_hits]

        reg_loss = self.regression_loss(
            influencers_predictions,
            influential_hits,
            batch,
            self.hparams["regression_targets"],
        )

        loss = infl_loss + reg_loss

        self.log_dict(
            {
                "train_loss": loss,
                "train_influencer_loss": infl_loss,
                "train_follower_follower_loss": sublosses["follower_follower_loss"],
                "train_follower_influencer_loss": sublosses["follower_influencer_loss"],
                "train_influencer_influencer_loss": sublosses[
                    "influencer_influencer_loss"
                ],
                "train_regression_loss": reg_loss,
            }
        )

        if torch.isnan(loss):
            print("Loss is nan")
            sys.exit()

        return loss

    def regression_loss(self, predictions, hits, batch, target_keys):
        # Build the regression targets
        regression_targets = []
        for target_key in target_keys:
            regression_targets.append(batch[target_key][hits])

        regression_targets = torch.stack(regression_targets, dim=1)

        # Calculate the loss as MSE
        loss = torch.nn.functional.mse_loss(predictions, regression_targets)

        return loss

    def shared_evaluation(self, batch, batch_idx):
        input_data = self.get_input_data(batch)
        self.start_validation_tracking()
        follower_embed, influencer_embed = self(input_data, batch.batch)

        try:
            follower_influencer_edges, follower_influencer_truth = self.get_training_edges(
                batch,
                follower_embed,
                influencer_embed,
                hnm=True,
                knn=500,
                batch_index=batch.batch,
            )
        except Exception:
            follower_influencer_edges, follower_influencer_truth = torch.empty(
                [2, 0], dtype=torch.int64, device=self.device
            ), torch.empty([0], dtype=torch.int64, device=self.device)
        try:
            follower_follower_edges, follower_follower_truth = self.get_training_edges(
                batch,
                follower_embed,
                follower_embed,
                hnm=True,
                knn=500,
                batch_index=batch.batch,
            )
        except Exception:
            follower_follower_edges, follower_follower_truth = torch.empty(
                [2, 0], dtype=torch.int64, device=self.device
            ), torch.empty([0], dtype=torch.int64, device=self.device)
        try:
            (
                influencer_influencer_edges,
                influencer_influencer_truth,
            ) = self.get_training_edges(
                batch,
                influencer_embed,
                influencer_embed,
                hnm=True,
                radius=self.hparams["influencer_margin"],
                knn=500,
                batch_index=batch.batch,
            )
        except Exception:
            influencer_influencer_edges, influencer_influencer_truth = torch.empty(
                [2, 0], dtype=torch.int64, device=self.device
            ), torch.empty([0], dtype=torch.int64, device=self.device)

        # Compute the total loss
        loss, sublosses = influencer_loss(
            follower_embed,
            influencer_embed,
            batch,
            follower_follower_edges,
            follower_follower_truth,
            follower_influencer_edges,
            follower_influencer_truth,
            influencer_influencer_edges,
            influencer_influencer_truth,
            follower_follower_weight=self.follower_follower_weight,
            follower_influencer_weight=self.follower_influencer_weight,
            influencer_influencer_weight=self.influencer_influencer_weight,
            follower_follower_pos_ratio=self.hparams["follower_follower_pos_ratio"],
            follower_influencer_neg_ratio=self.hparams["follower_influencer_neg_ratio"],
            follower_margin=self.hparams["margin"],
            influencer_margin=self.hparams["influencer_margin"],
            device=self.device,
            scatter_loss=self.hparams["scatter_loss"],
        )

        current_lr = self.optimizers().param_groups[0]["lr"]

        cluster_eff, cluster_pur = self.get_cluster_metrics(
            batch, follower_follower_edges, follower_follower_truth
        )
        represent_eff, represent_pur, represent_dup = self.get_representative_metrics(
            batch, follower_influencer_edges, follower_influencer_truth
        )
        tracking_eff, tracking_pur, tracking_dup = self.get_tracking_metrics(
            batch, follower_influencer_edges, follower_influencer_truth
        )

        with contextlib.suppress(Exception):
            self.log_dict(
                {
                    "val_loss": loss,
                    "cluster_eff": cluster_eff,
                    "cluster_pur": cluster_pur,
                    "represent_pur": represent_pur,
                    "represent_eff": represent_eff,
                    "represent_dup": represent_dup,
                    "lr": current_lr,
                    "follower_follower_loss": sublosses["follower_follower_loss"],
                    "follower_influencer_loss": sublosses["follower_influencer_loss"],
                    "influencer_influencer_loss": sublosses[
                        "influencer_influencer_loss"
                    ],
                    "tracking_eff": tracking_eff,
                    "tracking_fake_rate": 1 - tracking_pur,
                    "tracking_dup": tracking_dup,
                },
            )
        if batch_idx == 0:
            print(
                f"Rep eff: {represent_eff}, rep pur: {represent_pur}, rep dup: {represent_dup}"
            )
            print(f"Cluster eff: {cluster_eff}, cluster pur: {cluster_pur}")

            first_event = Batch.to_data_list(batch)[0]
            batch_mask = batch.batch == 0

            # self.log_embedding_plot(batch, follower_embed[pid_mask], spatial2=influencer_embed[pid_mask], uu_edges=follower_follower_edges, ui_edges=follower_influencer_edges, ii_edges=influencer_influencer_edges)
            self.log_embedding_plot(
                batch,
                follower_embed[batch_mask],
                spatial2=influencer_embed[batch_mask],
                uu_edges=follower_follower_edges[:, batch_mask[follower_follower_edges].all(dim=0)],
                ui_edges=follower_influencer_edges[
                    :, batch_mask[follower_influencer_edges].all(dim=0)
                ],
                ii_edges=influencer_influencer_edges[
                    :, batch_mask[influencer_influencer_edges].all(dim=0)
                ],
            )

        return {
            "val_loss": loss,
            "cluster_eff": cluster_eff,
            "cluster_pur": cluster_pur,
            "represent_pur": represent_pur,
            "represent_eff": represent_eff,
            "represent_dup": represent_dup,
            "lr": current_lr,
            "follower_follower_loss": sublosses["follower_follower_loss"],
            "follower_influencer_loss": sublosses["follower_influencer_loss"],
            "influencer_influencer_loss": sublosses["influencer_influencer_loss"],
            "follower_follower_edges": follower_follower_edges,
            "follower_follower_truth": follower_follower_truth,
            "follower_influencer_edges": follower_influencer_edges,
            "follower_influencer_truth": follower_influencer_truth,
            "influencer_influencer_edges": influencer_influencer_edges,
            "influencer_influencer_truth": influencer_influencer_truth,
            "follower_embed": follower_embed,
            "influencer_embed": influencer_embed,
        }

    # Abstract the above into a function
    def get_weight(self, weight_name):
        if not isinstance(self.hparams[weight_name], list):
            return self.hparams[weight_name]
        if len(self.hparams[weight_name]) == 2:
            start = self.hparams[weight_name][0]
            end = self.hparams[weight_name][1]
            return (
                start + (end - start) * self.current_epoch / self.hparams["max_epochs"]
            )
        elif len(self.hparams[weight_name]) == 3:
            start = self.hparams[weight_name][0]
            mid = self.hparams[weight_name][1]
            end = self.hparams[weight_name][2]
            return (
                start
                + (mid - start) * self.current_epoch / (self.hparams["max_epochs"] / 2)
                if self.current_epoch < self.hparams["max_epochs"] / 2
                else mid
                + (end - mid)
                * (self.current_epoch - self.hparams["max_epochs"] / 2)
                / (self.hparams["max_epochs"] / 2)
            )

    @property
    def follower_influencer_weight(self):
        return self.get_weight("follower_influencer_weight")

    @property
    def influencer_influencer_weight(self):
        return self.get_weight("influencer_influencer_weight")

    @property
    def follower_follower_weight(self):
        return self.get_weight("follower_follower_weight")
