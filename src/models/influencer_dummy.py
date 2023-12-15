# 3rd party imports
from .influencer_model import InfluencerModel
from torch import nn
import torch
import sys
import contextlib
from torch_geometric.data import Batch
import numpy as np

from ..losses.influencer_loss import influencer_loss

PRINT_RATE = 1000


class InfluencerDummy(InfluencerModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """

        self.setup(stage="fit")

        self.valset = self.trainset

        # Get the 6D positions of the nodes from the trainset and make them the trainable parameters of the model
        self.followers = nn.Parameter(self.trainset[0].x)
        self.influencers = nn.Parameter(self.trainset[0].x)

        self.save_hyperparameters()

    def forward(self, x, batch=None):
        return self.followers, self.influencers

    def training_step(self, batch, batch_idx):
        """
        The Influencer training step.
        1. Runs the model in no_grad mode to get the follower and influencer embeddings
        2. Builds hard negative, random pairs, and true edges for the follower-follower loss
        3. Build hard negatives and true edges for the follower-influencer loss
        4. Build true edges for the influencer-influencer loss
        5. Compute the loss
        """

        batch = list(self.train_dataloader())[0].to(self.device)

        # Get the follower and influencer embeddings
        input_data = self.get_input_data(batch)
        follower_embed, influencer_embed = self(input_data, batch.batch)

        # Get the training edges for each loss function
        follower_follower_edges, follower_follower_truth = self.get_training_edges(
            batch,
            follower_embed,
            follower_embed,
            hnm=True,
            rp=False,
            tp=True,
            batch_index=batch.batch,
        )
        follower_influencer_edges, follower_influencer_truth = self.get_training_edges(
            batch,
            follower_embed,
            influencer_embed,
            hnm=True,
            tp=True,
            rp=False,
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
            rp=False,
            radius=self.hparams["influencer_margin"],
            batch_index=batch.batch,
        )

        # Calculate each loss function
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
            loss_type=self.hparams["loss_type"],
        )

        self.log_dict(
            {
                "train_loss": loss,
                "train_follower_follower_loss": sublosses["follower_follower_loss"],
                "train_follower_influencer_loss": sublosses["follower_influencer_loss"],
                "train_influencer_influencer_loss": sublosses[
                    "influencer_influencer_loss"
                ],
            }
        )
        if self.current_epoch % PRINT_RATE == 0:
            print(
                f"Train loss: {loss}, follower-follower loss: {sublosses['follower_follower_loss']}, follower-influencer loss: {sublosses['follower_influencer_loss']}, influencer-influencer loss: {sublosses['influencer_influencer_loss']}"
            )

        if torch.isnan(loss):
            print("Loss is nan")
            sys.exit()

        return loss

    def shared_evaluation(self, batch, batch_idx, val_radius=None):
        if val_radius is None:
            val_radius = self.hparams["val_radius"]

        print("VAL_RADIUS = ", val_radius)

        batch = list(self.train_dataloader())[0].to(self.device)

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
                radius=val_radius,
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
                radius=val_radius,
                self_loop=True,
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
            loss_type=self.hparams["loss_type"],
        )

        # Calculate average distances between each edges for true and fake separately
        true_follower_influencer_edges = follower_influencer_edges[
            :, follower_influencer_truth.nonzero().squeeze()
        ]
        fake_follower_influencer_edges = follower_influencer_edges[
            :, (~follower_influencer_truth).nonzero().squeeze()
        ]
        follower_influencer_distances_true = (
            follower_embed[true_follower_influencer_edges[0]]
            - influencer_embed[true_follower_influencer_edges[1]]
        ).norm(dim=-1)
        follower_influencer_distances_fake = (
            follower_embed[fake_follower_influencer_edges[0]]
            - influencer_embed[fake_follower_influencer_edges[1]]
        ).norm(dim=-1)
        avg_true_follower_influencer_dist = follower_influencer_distances_true.mean()
        avg_fake_follower_influencer_dist = follower_influencer_distances_fake.mean()
        median_true_follower_influencer_dist = follower_influencer_distances_true.median()
        median_fake_follower_influencer_dist = follower_influencer_distances_fake.median()
        hist_true_follower_influencer_dist = torch.histc(
            follower_influencer_distances_true, bins=10, min=0, max=1
        )
        hist_fake_follower_influencer_dist = torch.histc(
            follower_influencer_distances_fake, bins=10, min=0, max=1
        )

        true_influencer_influencer_edges = influencer_influencer_edges[
            :, influencer_influencer_truth.nonzero().squeeze()
        ]
        fake_influencer_influencer_edges = influencer_influencer_edges[
            :, (~influencer_influencer_truth).nonzero().squeeze()
        ]
        influencer_influencer_distances_true = (
            influencer_embed[true_influencer_influencer_edges[0]]
            - influencer_embed[true_influencer_influencer_edges[1]]
        ).norm(dim=-1)
        influencer_influencer_distances_fake = (
            influencer_embed[fake_influencer_influencer_edges[0]]
            - influencer_embed[fake_influencer_influencer_edges[1]]
        ).norm(dim=-1)
        avg_true_influencer_influencer_dist = (
            influencer_influencer_distances_true.mean()
        )
        avg_fake_influencer_influencer_dist = (
            influencer_influencer_distances_fake.mean()
        )
        median_true_influencer_influencer_dist = (
            influencer_influencer_distances_true.median()
        )
        median_fake_influencer_influencer_dist = (
            influencer_influencer_distances_fake.median()
        )
        hist_true_influencer_influencer_dist = torch.histc(
            influencer_influencer_distances_true, bins=10, min=0, max=1
        )
        hist_fake_influencer_influencer_dist = torch.histc(
            influencer_influencer_distances_fake, bins=10, min=0, max=1
        )

        true_follower_follower_edges = follower_follower_edges[:, follower_follower_truth.nonzero().squeeze()]
        fake_follower_follower_edges = follower_follower_edges[
            :, (~follower_follower_truth).nonzero().squeeze()
        ]
        follower_follower_distances_true = (
            follower_embed[true_follower_follower_edges[0]] - follower_embed[true_follower_follower_edges[1]]
        ).norm(dim=-1)
        follower_follower_distances_fake = (
            follower_embed[fake_follower_follower_edges[0]] - follower_embed[fake_follower_follower_edges[1]]
        ).norm(dim=-1)
        avg_true_follower_follower_dist = follower_follower_distances_true.mean()
        avg_fake_follower_follower_dist = follower_follower_distances_fake.mean()
        median_true_follower_follower_dist = follower_follower_distances_true.median()
        median_fake_follower_follower_dist = follower_follower_distances_fake.median()
        hist_true_follower_follower_dist = torch.histc(
            follower_follower_distances_true, bins=10, min=0, max=1
        )
        hist_fake_follower_follower_dist = torch.histc(
            follower_follower_distances_fake, bins=10, min=0, max=1
        )

        # Print the average distances, median distances and histogram binning
        # print(f"Average true follower-follower distance: {avg_true_follower_follower_dist}")
        # print(f"Average fake follower-follower distance: {avg_fake_follower_follower_dist}")
        # print(f"Average true follower-influencer distance: {avg_true_follower_influencer_dist}")
        # print(f"Average fake follower-influencer distance: {avg_fake_follower_influencer_dist}")
        # print(f"Average true influencer-influencer distance: {avg_true_influencer_influencer_dist}")
        # print(f"Average fake influencer-influencer distance: {avg_fake_influencer_influencer_dist}")
        # print(f"Median true follower-follower distance: {median_true_follower_follower_dist}")
        # print(f"Median fake follower-follower distance: {median_fake_follower_follower_dist}")
        # print(f"Median true follower-influencer distance: {median_true_follower_influencer_dist}")
        # print(f"Median fake follower-influencer distance: {median_fake_follower_influencer_dist}")
        # print(f"Median true influencer-influencer distance: {median_true_influencer_influencer_dist}")
        # print(f"Median fake influencer-influencer distance: {median_fake_influencer_influencer_dist}")
        # print(f"Histogram binning of true follower-follower distances: {hist_true_follower_follower_dist}")
        # print(f"Histogram binning of fake follower-follower distances: {hist_fake_follower_follower_dist}")
        # print(f"Histogram binning of true follower-influencer distances: {hist_true_follower_influencer_dist}")
        # print(f"Histogram binning of fake follower-influencer distances: {hist_fake_follower_influencer_dist}")
        # print(f"Histogram binning of true influencer-influencer distances: {hist_true_influencer_influencer_dist}")
        # print(f"Histogram binning of fake influencer-influencer distances: {hist_fake_influencer_influencer_dist}")

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
            "cluster_eff": cluster_eff,
            "cluster_pur": cluster_pur,
            "represent_pur": represent_pur,
            "represent_eff": represent_eff,
            "represent_dup": represent_dup,
            "tracking_eff": tracking_eff,
            "tracking_fake_rate": 1 - tracking_pur,
            "tracking_dup": tracking_dup,
        }

    def validation_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """

        if self.current_epoch % PRINT_RATE == 0:
            print("-------------------------------------------")
            print("Validation step")
            print(f"Current learning rate: {self.optimizers().param_groups[0]['lr']}")

            for val_radius in np.arange(0, self.hparams["val_radius"], 0.1):
                self.shared_evaluation(batch, batch_idx, val_radius=val_radius)

            return self.shared_evaluation(batch, batch_idx)
