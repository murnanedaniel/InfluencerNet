import contextlib

# System imports
import sys
import pandas as pd

import torch

# Local Imports
from .object_condensation_base import ObjectCondensationBase
from .utils import build_edges
from torch_geometric.data import Dataset, Batch
from torch_geometric.nn import aggr
import copy
import numpy as np

from ..losses.influencer_loss import influencer_loss

device = "cuda" if torch.cuda.is_available() else "cpu"
sqrt_eps = 1e-12

PRINT_RATE = 1


class InfluencerModel(ObjectCondensationBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """
        self.save_hyperparameters(hparams)
        self.original_pca = None
        self.embedding_pca = None
        self.mean_agg = aggr.MeanAggregation()
        self.val_radius_history = (
            []
        )  # Used to track the best validation radius over time

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
            try:
                training_edges = self.append_hnm_pairs(
                    training_edges,
                    embedding_query,
                    embedding_database,
                    radius=radius,
                    knn=knn,
                    batch_index=batch_index,
                    self_loop=self_loop,
                )
            except Exception:
                pass

        # Append Random Pairs (rp) with KNN graph
        if rp:
            try:
                training_edges = self.append_random_pairs(
                    training_edges,
                    embedding_query,
                    embedding_database,
                    batch_index=batch_index,
                )
            except Exception:
                pass

        # Append True Pairs (tp) with KNN graph
        if tp:
            training_edges = self.append_true_pairs(training_edges, batch)

        # Remove duplicates
        training_edges = training_edges.unique(dim=1)

        # Get the truth values for the edges
        training_truth = self.get_truth(training_edges, batch)

        return training_edges, training_truth

    def training_step(self, batch, batch_idx):
        """
        The Influencer training step.
        1. Runs the model in no_grad mode to get the user and influencer embeddings
        2. Builds hard negative, random pairs, and true edges for the user-user loss
        3. Build hard negatives and true edges for the user-influencer loss
        4. Build true edges for the influencer-influencer loss
        5. Compute the loss
        """

        # Get the user and influencer embeddings
        input_data = self.get_input_data(batch)
        with torch.no_grad():
            user_embed, influencer_embed = self(input_data, batch=batch.batch)

        # Get the training edges for each loss function
        user_influencer_edges, user_influencer_truth = self.get_training_edges(
            batch,
            user_embed,
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
            [user_influencer_edges, influencer_influencer_edges], dim=1
        ).unique()
        user_embed[included_hits], influencer_embed[included_hits] = self(
            input_data[included_hits], batch=batch.batch[included_hits]
        )

        # Calculate each loss function

        loss, sublosses = influencer_loss(
            user_embed,
            influencer_embed,
            batch,
            user_influencer_edges,
            user_influencer_truth,
            influencer_influencer_edges,
            influencer_influencer_truth,
            user_influencer_weight=self.user_influencer_weight,
            influencer_influencer_weight=self.influencer_influencer_weight,
            user_influencer_neg_ratio=self.hparams["user_influencer_neg_ratio"],
            user_margin=self.hparams["margin"],
            influencer_margin=self.hparams["influencer_margin"],
            device=self.device,
            scatter_loss=self.hparams.get("scatter_loss", True),
            loss_type=self.hparams.get("loss_type", "hinge"),
        )

        self.log_dict(
            {
                "train_loss": loss,
                "train_user_influencer_loss": sublosses["user_influencer_loss"],
                "train_influencer_influencer_loss": sublosses[
                    "influencer_influencer_loss"
                ],
            },
            on_epoch=True,
            on_step=False,
        )

        if torch.isnan(loss):
            print("Loss is nan")
            sys.exit()

        return loss

    def shared_evaluation(self, batch, batch_idx, val_radius=None):
        if val_radius is None:
            if self.moving_average_val_radius == 0:
                print(self.hparams["val_radius"])
                val_radius = self.hparams.get("val_radius", self.hparams["margin"])
            else:
                val_radius = self.moving_average_val_radius

        print("VAL_RADIUS = ", val_radius)

        input_data = self.get_input_data(batch)
        self.start_validation_tracking()
        user_embed, influencer_embed = self(input_data, batch=batch.batch)

        try:
            user_influencer_edges, user_influencer_truth = self.get_training_edges(
                batch,
                user_embed,
                influencer_embed,
                hnm=True,
                knn=500,
                batch_index=batch.batch,
                radius=val_radius,
            )
        except Exception:
            user_influencer_edges, user_influencer_truth = torch.empty(
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
            user_embed,
            influencer_embed,
            batch,
            user_influencer_edges,
            user_influencer_truth,
            influencer_influencer_edges,
            influencer_influencer_truth,
            user_influencer_weight=self.user_influencer_weight,
            influencer_influencer_weight=self.influencer_influencer_weight,
            user_influencer_neg_ratio=self.hparams["user_influencer_neg_ratio"],
            user_margin=self.hparams["margin"],
            influencer_margin=self.hparams["influencer_margin"],
            device=self.device,
            scatter_loss=self.hparams["scatter_loss"],
            loss_type=self.hparams.get("loss_type", "hinge"),
        )

        try:
            current_lr = (
                self.optimizers().param_groups[0]["lr"] if self.optimizers() else 0
            )
        except Exception:
            current_lr = None

        represent_eff, represent_pur, represent_dup = self.get_representative_metrics(
            batch, user_influencer_edges, user_influencer_truth
        )
        tracking_eff, tracking_pur, tracking_dup = self.get_tracking_metrics(
            batch, user_influencer_edges, user_influencer_truth
        )

        f1 = tracking_eff * tracking_pur * (1 - tracking_dup)

        return {
            "val_loss": loss,
            "represent_pur": represent_pur,
            "represent_eff": represent_eff,
            "represent_dup": represent_dup,
            "lr": current_lr,
            "user_influencer_loss": sublosses["user_influencer_loss"],
            "influencer_influencer_loss": sublosses["influencer_influencer_loss"],
            "user_influencer_edges": user_influencer_edges,
            "user_influencer_truth": user_influencer_truth,
            "influencer_influencer_edges": influencer_influencer_edges,
            "influencer_influencer_truth": influencer_influencer_truth,
            "user_embed": user_embed,
            "influencer_embed": influencer_embed,
            "tracking_eff": tracking_eff,
            "tracking_pur": tracking_pur,
            "tracking_dup": tracking_dup,
            "f1": f1,
        }

    def validation_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """

        if self.current_epoch % PRINT_RATE == 0:
            print("-------------------------------------------")
            print("Validation step")
            print(f"Current learning rate: {self.optimizers().param_groups[0]['lr']}")

            best_f1 = 0
            best_val_radius = None
            best_outputs = None
            for val_radius in np.arange(0, self.hparams.get("val_radius", 1.0), 0.1):
                outputs = self.shared_evaluation(
                    batch, batch_idx, val_radius=val_radius
                )
                if outputs["f1"] >= best_f1:
                    best_f1 = outputs["f1"]
                    best_outputs = outputs
                    best_val_radius = val_radius

            self.val_radius_history.append(best_val_radius)
            if len(self.val_radius_history) > 100:  # Replace N with the desired size
                self.val_radius_history.pop(0)

            self.log_dict(
                {
                    "val_loss": best_outputs["val_loss"],
                    "represent_pur": best_outputs["represent_pur"],
                    "represent_eff": best_outputs["represent_eff"],
                    "represent_dup": best_outputs["represent_dup"],
                    "lr": best_outputs["lr"],
                    "user_influencer_loss": best_outputs["user_influencer_loss"],
                    "influencer_influencer_loss": best_outputs[
                        "influencer_influencer_loss"
                    ],
                    "tracking_eff": best_outputs["tracking_eff"],
                    "tracking_fake_rate": 1 - best_outputs["tracking_pur"],
                    "tracking_dup": best_outputs["tracking_dup"],
                    "f1": best_outputs["f1"],
                    "best_val_radius": best_val_radius,
                }
            )

            return best_outputs

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
    def user_influencer_weight(self):
        return self.get_weight("user_influencer_weight")

    @property
    def influencer_influencer_weight(self):
        return self.get_weight("influencer_influencer_weight")

    @property
    def user_user_weight(self):
        return self.get_weight("user_user_weight")

    @property
    def moving_average_val_radius(self):
        if len(self.val_radius_history) == 0:
            return 0
        return sum(self.val_radius_history) / len(self.val_radius_history)
