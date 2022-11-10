"""The base classes for the embedding process.

The embedding process here is both the embedding models (contained in Models/) and the training procedure, which is a Siamese network strategy. Here, the network is run on all points, and then pairs (or triplets) are formed by one of several strategies (e.g. random pairs (rp), hard negative mining (hnm)) upon which some sort of contrastive loss is applied. The default here is a hinge margin loss, but other loss functions can work, including cross entropy-style losses. Also available are triplet loss approaches.

Example:
    See Quickstart for a concrete example of this process.
    
Todo:
    * Refactor the training & validation pipeline, since the use of the different regimes (rp, hnm, etc.) looks very messy
"""

# System imports
import sys
import os
import logging
import warnings

# 3rd party imports
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torch
from torch.nn import Linear
from torch_geometric.data import DataLoader
from torch_cluster import radius_graph
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import io
import wandb
from sklearn.decomposition import PCA
import plotly.graph_objects as go

# Local Imports
from .generation_utils import graph_intersection, build_dataset
from .utils import build_edges

device = "cuda" if torch.cuda.is_available() else "cpu"


class EmbeddingBase(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """
        self.save_hyperparameters(hparams)
        self.trainset, self.valset, self.testset = None, None, None

    def setup(self, stage="fit"):
        if not self.trainset or not self.valset or not self.testset:
            self.trainset, self.valset, self.testset = build_dataset(**self.hparams)

        try:
            self.logger.experiment.define_metric("val_loss", summary="min")
            self.log_embedding_plot(self.valset[0], spatial1=self.valset[0].x, uu_edges=self.valset[0].all_signal_edges)
        except Exception:
            warnings.warn("Could not define metrics for W&B")

    def train_dataloader(self):
        if self.trainset is not None:
            return DataLoader(self.trainset, batch_size=1, num_workers=0)
        else:
            return None

    def val_dataloader(self):
        if self.valset is not None:
            return DataLoader(self.valset, batch_size=1, num_workers=0, shuffle=False)
        else:
            return None

    def test_dataloader(self):
        if self.testset is not None:
            return DataLoader(self.testset, batch_size=1, num_workers=1)
        else:
            return None

    def configure_optimizers(self):
        optimizer = [
            torch.optim.AdamW(
                self.parameters(),
                lr=(self.hparams["lr"]),
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=True,
            )
        ]
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer[0],
                    step_size=self.hparams["patience"],
                    gamma=self.hparams["factor"],
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return optimizer, scheduler

    def training_step(self, batch, batch_idx):

        training_edges, training_truth, embedded_nodes = self.get_training_edges(batch)
        loss = self.get_loss(training_edges, training_truth, batch, embedded_nodes)

        self.log("train_loss", loss)

        return loss

    def shared_evaluation(self, batch, batch_idx):

        validation_edges, validation_truth, spatial = self.get_validation_edges(batch)

        loss = self.get_loss(validation_edges, validation_truth, batch)

        current_lr = self.optimizers().param_groups[0]["lr"]

        cluster_true = batch.all_signal_edges.shape[1]
        cluster_true_positive = validation_truth.sum()
        cluster_positive = validation_edges.shape[1]

        eff = cluster_true_positive / cluster_true
        pur = cluster_true_positive / cluster_positive

        self.log_dict(
            {"val_loss": loss, "eff": eff, "pur": pur, "current_lr": current_lr}
        )

        logging.info(f"Efficiency: {eff}")
        logging.info(f"Purity: {pur}")

        if batch_idx == 0:
            self.log_embedding_plot(batch, spatial, uu_edges=validation_edges)

        return loss

    def get_input_data(self, batch):

        input_data = batch.x
        input_data[input_data != input_data] = 0

        return input_data    

    def append_true_pairs(self, edges, batch):

        edges = torch.cat([edges.to(self.device), batch.all_signal_edges.to(self.device)], dim=-1)

        return edges
                
    def get_hinge_distance(self, query, database, edges, truth):

        hinge = truth.float().to(self.device)
        hinge[hinge == 0] = -1
        d = torch.sum((query[edges[0]] - database[edges[1]]) ** 2, dim=-1)

        return hinge, d

    def get_truth(self, edges, batch):

        return batch.pid[edges[0]] == batch.pid[edges[1]]

    def remove_duplicate_edges(self, edges):

        edges = torch.cat([edges, edges.flip(0)], dim=-1)
        edges = torch.unique(edges, dim=-1)

        return edges

    def get_query_points(self, batch, spatial):

        query_indices = batch.all_signal_edges.unique()
        query = spatial[query_indices]

        return query_indices, query

    def get_training_edges(self, batch):
                
        # Instantiate empty prediction edge list
        training_edges = torch.empty([2, 0], dtype=torch.int64, device=self.device)

        # Forward pass of model, handling whether Cell Information (ci) is included
        input_data = self.get_input_data(batch)

        with torch.no_grad():
            spatial = self(input_data)

        query_indices, query = self.get_query_points(batch, spatial)

        # Append Hard Negative Mining (hnm) with KNN graph
        if "hnm" in self.hparams["regime"]:
            training_edges = self.append_hnm_pairs(training_edges, query, spatial, query_indices)

        # Append random edges pairs (rp) for stability
        if "rp" in self.hparams["regime"]:
            training_edges = self.append_random_pairs(training_edges, spatial, query_indices)

        # Append all positive examples and their truth and weighting
        training_edges = self.append_true_pairs(training_edges, batch)
        training_edges = self.remove_duplicate_edges(training_edges)
        training_truth = self.get_truth(training_edges, batch)

        return training_edges, training_truth, spatial

    def get_loss(self, edges, truth, batch, spatial=None):

        if spatial is not None:
            included_hits = edges.unique()
            spatial[included_hits] = self(self.get_input_data(batch)[included_hits])
        else:
            spatial = self(self.get_input_data(batch))

        hinge, d = self.get_hinge_distance(spatial, spatial, edges, truth)

        negative_loss = torch.nn.functional.hinge_embedding_loss(
            d[hinge == -1],
            hinge[hinge == -1],
            margin=self.hparams["margin"]**2,
            reduction="mean",
        )

        positive_loss = torch.nn.functional.hinge_embedding_loss(
            d[hinge == 1],
            hinge[hinge == 1],
            margin=self.hparams["margin"]**2,
            reduction="mean",
        )

        loss = negative_loss + self.hparams["weight"] * positive_loss

        # Check if loss is nan and exit
        if torch.isnan(loss):
            print("Loss is nan")
            print(edges)
            print(truth)
            print(spatial)
            print(torch.isnan(spatial).any(), torch.isnan(edges).any(), torch.isnan(truth).any())
            print(torch.isnan(hinge), torch.isnan(d))
            sys.exit()

        return loss


    def get_validation_edges(self, batch):

        input_data = self.get_input_data(batch)
        spatial = self(input_data)

        # Build whole KNN graph
        edges = build_edges(
            spatial, spatial, indices=None, r_max=self.hparams["radius"], k_max=500
        )

        truth = self.get_truth(edges, batch)
        edges, truth = edges.to(self.device), truth.to(self.device)

        return edges, truth, spatial

    def append_hnm_pairs(self, edges, query, database, query_indices=None, radius=None, knn=None):

        if radius is None:
            radius = self.hparams["radius"]
        if knn is None:
            knn = self.hparams["knn"]

        if query_indices is None:
            query_indices = torch.arange(len(query), device=self.device)

        knn_edges = build_edges(
            query,
            database,
            query_indices,
            r_max=radius,
            k_max=knn,
        )

        edges = torch.cat([edges, knn_edges], dim=-1)

        return edges

    def append_random_pairs(self, edges, database, query_indices=None):

        if query_indices is None:
            query_indices = torch.arange(len(database), device=self.device)

        n_random = int(self.hparams["randomisation"] * len(query_indices))
        indices_src = torch.randint(0, len(query_indices), (n_random,), device=self.device)
        indices_dest = torch.randint(0, len(database), (n_random,), device=self.device)
        random_pairs = torch.stack([query_indices[indices_src], indices_dest])

        edges = torch.cat([edges, random_pairs], dim=-1)

        return edges

    def validation_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """
        return self.shared_evaluation(
            batch, batch_idx
        )

    def test_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """
        return self.shared_evaluation(batch, batch_idx)

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure=None,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        """
        Use this to manually enforce warm-up. In the future, this may become built-into PyLightning
        """
        logging.info(f"Optimizer step for batch {batch_idx}")
        # warm up lr
        if (self.hparams["warmup"] is not None) and (
            self.trainer.current_epoch < self.hparams["warmup"]
        ):
            lr_scale = min(
                1.0, float(self.trainer.current_epoch + 1) / self.hparams["warmup"]
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams["lr"]

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

        logging.info(f"Optimizer step done for batch {batch_idx}")

    def plot_embedding_image(self, batch, validation_edges, spatial):

        fig = self.plot_embedding_native(batch, validation_edges, spatial)

        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return wandb.Image(data)

    def log_embedding_plot(self, batch, spatial1, uu_edges=None, ui_edges=None, ii_edges=None, spatial2=None, label="embeddings"):

        if self.logger.experiment is not None:
            self.logger.experiment.log(
                {
                label: self.plot_embedding_native(batch, spatial1, spatial2, uu_edges, ui_edges, ii_edges)
                }
            )

    def plot_embedding_native(self, batch, spatial1, spatial2=None, uu_edges=None, ui_edges=None, ii_edges=None):

        fig = go.Figure()

        if spatial2 is None:
            spatial1_cpu, spatial2_cpu = spatial1.cpu(), spatial1.cpu()
        else:
            spatial1_cpu, spatial2_cpu = spatial1.cpu(), spatial2.cpu()
        

        if uu_edges is not None and uu_edges.shape[1] > 0:
            self.plot_edges(fig, uu_edges, spatial1_cpu, spatial1_cpu, batch.pid, color1="green", color2="orange")

        if ii_edges is not None and ii_edges.shape[1] > 0:
            self.plot_edges(fig, ii_edges, spatial2_cpu, spatial2_cpu, batch.pid, color1="purple", color2="grey")

        if ui_edges is not None and ui_edges.shape[1] > 0:
            self.plot_edges(fig, ui_edges, spatial1_cpu, spatial2_cpu, batch.pid, color1="blue", color2="red")

        # Plot spatial1 points as circles, with each point colored, with a map of PID to color
        fig.add_trace(
            go.Scatter(
                x=spatial1_cpu[:, 0],
                y=spatial1_cpu[:, 1],
                mode="markers",
                marker=dict(
                    size=10,
                    color=batch.pid.cpu(),
                )
            )
        )

        if spatial2 is not None:
            # Plot spatial2 points as stars, with each point colored, with a map of PID to color
            fig.add_trace(
                go.Scatter(
                    x=spatial2_cpu[:, 0],
                    y=spatial2_cpu[:, 1],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=batch.pid.cpu(),
                        symbol="star"
                        )
                    )
                )
            if ui_edges is not None and ui_edges.shape[1] > 0:
                for pid in batch.pid.unique():
                    rep_with_pid = ui_edges[1, batch.pid[ui_edges[1]] == pid].unique().cpu()

                    fig.add_trace(
                        go.Scatter(
                            x=spatial2_cpu[rep_with_pid, 0],
                            y=spatial2_cpu[rep_with_pid, 1],
                            mode="markers",
                            marker=dict(
                                size=14,
                                color=batch.pid[rep_with_pid].cpu(),
                                symbol="star"
                                )
                            )
                        )

        return fig

    def plot_edges(self, fig, edges, spatial1, spatial2, pid, color1="blue", color2="red"):
            
            true_edges = pid[edges[0]] == pid[edges[1]]
            for edge in edges[:, true_edges].cpu().T:
                fig.add_trace(
                    go.Scatter(
                        x=[spatial1[edge[0], 0], spatial2[edge[1], 0]],
                        y=[spatial1[edge[0], 1], spatial2[edge[1], 1]],
                        mode="lines",
                        line=dict(color=color1, width=1),
                        showlegend=False,
                    )
                )
            for edge in edges[:, ~true_edges].cpu().T:
                fig.add_trace(
                    go.Scatter(
                        x=[spatial1[edge[0], 0], spatial2[edge[1], 0]],
                        y=[spatial1[edge[0], 1], spatial2[edge[1], 1]],
                        mode="lines",
                        line=dict(color=color2, width=1),
                        showlegend=False,
                    )
                )