# System imports
import sys
import logging
import warnings
import time

# 3rd party imports
import lightning.pytorch as pl
import torch.optim.lr_scheduler as lr_scheduler
from lightning.pytorch.utilities import grad_norm
import torch
from torch_geometric.data import DataLoader
from torch_geometric.nn import radius, knn
import numpy as np
import matplotlib.pyplot as plt
import wandb
from sklearn.decomposition import IncrementalPCA
import plotly.graph_objects as go
import pandas as pd
import math

from .. import datasets
from .. import networks
from .. import losses
from .. import utils
from .utils import build_edges

import class_resolver
import torch_geometric as pyg

# Global definitions
device = "cuda" if torch.cuda.is_available() else "cpu"
sqrt_eps = 1e-12
dataset_resolver = class_resolver.Resolver(classes = [datasets.ToyDataset, datasets.TrackMLDataset, datasets.RegressionDataset], base = pyg.data.Dataset)
model_resolver = class_resolver.Resolver(classes = [networks.NaiveTransformer, networks.InfluencerTransformer, networks.RegressionTransformer, networks.RegressionTransformerPyG, networks.RegressionInteractionGNN, networks.BinnedRegressionInteractionGNN], base = torch.nn.Module)

class ObjectCondensationBase(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """
        self.save_hyperparameters(hparams)
        self.trainset, self.valset, self.testset = None, None, None
        self.dataset_class = dataset_resolver.lookup(hparams["dataset"])
        self.network = model_resolver.lookup(hparams.get("network", None))(hparams)

    def forward(self, x, **kwargs):
        return self.network(x, **kwargs)

    def setup(self, stage="fit"):
        print("Setting up the data...")
        if not self.trainset or not self.valset or not self.testset:
            for data_name, data_num in zip(["trainset", "valset", "testset"], self.hparams["data_split"]):
                if data_num > 0:
                    input_dir = self.hparams["input_dir"] if "input_dir" in self.hparams else None
                    dataset = self.dataset_class(num_events=data_num, hparams=self.hparams, data_name = data_name, input_dir=input_dir)
                    setattr(self, data_name, dataset)

        try:
            self.logger.experiment.define_metric("val_loss", summary="min")
            self.log_embedding_plot(self.valset[0], spatial1=self.valset[0].x, uu_edges=self.valset[0].edge_index)
        except Exception:
            warnings.warn("Could not define metrics for W&B")

    def train_dataloader(self):
        if self.trainset is not None:
            return DataLoader(self.trainset, batch_size=self.hparams["batch_size"], num_workers=8, shuffle=False)
        else:
            return None

    def val_dataloader(self):
        if self.valset is not None:
            return DataLoader(self.valset, batch_size=self.hparams["batch_size"], num_workers=8, shuffle=False)
        else:
            return None

    def test_dataloader(self):
        if self.testset is not None:
            return DataLoader(self.testset, batch_size=self.hparams["batch_size"], num_workers=1)
        else:
            return None

    def make_lr_scheduler(self, optimizer):
        warmup_epochs = self.hparams["warmup"]
        lr_decay_factor = self.hparams["factor"]
        patience = self.hparams["patience"]
        T_max = self.hparams["max_epochs"]  # Maximum number of epochs
        scheduler_type = self.hparams.get("scheduler", "lambda")

        if scheduler_type == "lambda":
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    # During warm-up, increase the learning rate linearly
                    return (epoch + 1) / warmup_epochs
                else:
                    # After warm-up, decay the learning rate by lr_decay_factor every 10 epochs
                    return lr_decay_factor ** (epoch // patience)
        elif scheduler_type == "cosine":
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    # During warm-up, increase the learning rate linearly
                    return (epoch + 1) / warmup_epochs
                else:
                    # After warm-up, use cosine annealing
                    return 0.5 * (1 + math.cos(math.pi * epoch / T_max))
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
            
        return lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def on_before_optimizer_step(self, optimizer):
        # norms = grad_norm(self.layer, norm_type=2)
        # Calculate norms of all layers in model
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)

    def configure_optimizers(self):
        optimizer_type = self.hparams.get("optimizer", "AdamW")
        if optimizer_type == "AdamW":
            optimizer = [
                torch.optim.AdamW(
                    self.parameters(),
                    lr=self.hparams["lr"],
                    betas=(0.9, 0.999),
                    eps=1e-08,
                    amsgrad=False, 
                )
            ]
        elif optimizer_type == "SGD":
            optimizer = [
                torch.optim.SGD(
                    self.parameters(),
                    lr=self.hparams["lr"],
                    momentum=0.9,
                )
            ]
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        scheduler = [
            {
                "scheduler": self.make_lr_scheduler(optimizer[0]),
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
            }
        ]
        return optimizer, scheduler

    def training_step(self, batch, batch_idx):

        training_edges, training_truth, embedded_nodes = self.get_training_edges(batch)
        loss = self.get_loss(training_edges, training_truth, batch, embedded_nodes)

        self.log("train_loss", loss)

        return loss
    
    def on_train_batch_start(self, batch, batch_idx):
        """
        Want to track the max memory and time cost of a training epoch
        """
        self.training_tic = self.start_tracking()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        Want to track the memory and time cost of a training epoch
        """

        self.training_toc = time.time()
        self.log("stats/training_time", self.training_toc - self.training_tic)
        self.log("stats/training_max_memory", torch.cuda.max_memory_allocated() / 1024 ** 3)
        
    def start_validation_tracking(self):
        """
        Want to track the max memory and time cost of a validation epoch
        """
        self.validation_tic = self.start_tracking()

    def start_tracking(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        return time.time()

    def end_validation_tracking(self):
        """
        Want to track the memory and time cost of a validation epoch
        """
        self.validation_toc = time.time()
        self.log("stats/validation_time", self.validation_toc - self.validation_tic)
        self.log("stats/validation_max_memory", torch.cuda.max_memory_allocated()/ 1024 ** 3)

    def shared_evaluation(self, batch, batch_idx):

        validation_edges, validation_truth, spatial = self.get_validation_edges(batch)

        loss = self.get_loss(validation_edges, validation_truth, batch)

        current_lr = self.optimizers().param_groups[0]["lr"]

        cluster_true = batch.edge_index.shape[1]
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

        edges = torch.cat([edges.to(self.device), batch.edge_index.to(self.device)], dim=-1)

        return edges
                
    def get_hinge_distance(self, query, database, edges, truth, p=1):

        hinge = truth.float().to(self.device)
        hinge[hinge == 0] = -1
        if p==1:
            d = torch.sqrt(torch.sum((query[edges[0]] - database[edges[1]]) ** 2, dim=-1) + sqrt_eps) # EUCLIDEAN
        elif p==2:
            d = torch.sum((query[edges[0]] - database[edges[1]]) ** 2, dim=-1) # SQR-EUCLIDEAN
        else:
            raise NotImplementedError
        
        return hinge, d

    def get_truth(self, edges, batch):

        return batch.pid[edges[0]] == batch.pid[edges[1]]

    def remove_duplicate_edges(self, edges):

        edges = torch.cat([edges, edges.flip(0)], dim=-1)
        edges = torch.unique(edges, dim=-1)

        return edges

    def get_query_points(self, batch, spatial):

        query_indices = batch.edge_index.unique()
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

    def append_hnm_pairs(self, edges, query, database, query_indices=None, radius=None, knn=None, batch_index=None, self_loop=None):

        if radius is None:
            radius = self.hparams["radius"]
        if knn is None:
            knn = self.hparams["knn"]

        if query_indices is None:
            query_indices = torch.arange(len(query), device=self.device)

        if self_loop is None:
            if torch.equal(query, database):
                self_loop = False
            else:
                self_loop = True

        knn_edges = build_edges(
            query,
            database,
            query_indices,
            r_max=radius,
            k_max=knn,
            self_loop=self_loop,
            batch_index=batch_index,
        )

        edges = torch.cat([edges, knn_edges], dim=-1)

        return edges

    def append_random_pairs(self, edges, query, database, query_indices=None, batch_index=None):

        if batch_index is None:
            if query_indices is None:
                query_indices = torch.arange(len(database), device=self.device)

            n_random = int(self.hparams["randomisation"] * len(query_indices))
            indices_src = torch.randint(0, len(query_indices), (n_random,), device=self.device)
            indices_dest = torch.randint(0, len(database), (n_random,), device=self.device)
            random_pairs = torch.stack([query_indices[indices_src], indices_dest])
        else:
            # Simulate randomness by simply taking a KNN of 1 for each point
            random_pairs = knn(database, query, k = 2, batch_x = batch_index, batch_y = batch_index)
            # Remove self-edges
            random_pairs = random_pairs[:, random_pairs[0] != random_pairs[1]]

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
        Step to test the model's performance. Should do the following:
        1. Run the shared evaluation and return
        """
        if not hasattr(self, 'test_outputs'):
            self.test_outputs = []
        self.test_outputs.append(self.shared_evaluation(batch, batch_idx))

    def on_test_epoch_end(self):
        """
        End of test epoch. Should do the following:
        1. Print out the mean fake rate, efficiency, duplicate rate and f1 score
        2. Make a plot of efficiency vs. pT
        """

        # Gather each metric
        test_fake_rate = np.mean([1-output["tracking_pur"] for output in self.test_outputs])
        test_efficiency = np.mean([output["tracking_eff"] for output in self.test_outputs])
        test_duplicate_rate = np.mean([output["tracking_dup"] for output in self.test_outputs])
        test_f1_score = (1-test_fake_rate) * test_efficiency * (1-test_duplicate_rate)

        # Print out
        print(f"Test fake rate: {test_fake_rate}, test efficiency: {test_efficiency}, test duplicate rate: {test_duplicate_rate}, test f1 score: {test_f1_score}")

        # Make a plot of efficiency vs. pT
        # TODO: Make this plot!
        # utils.plotting.plot_efficiency_vs_pt(self.test_outputs)

    def on_train_start(self):
        self.optimizers().param_groups = self.optimizers()._optimizer.param_groups

    def plot_embedding_image(self, batch, validation_edges, spatial):

        fig = self.plot_embedding_native(batch, validation_edges, spatial)

        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return wandb.Image(data)

    def log_embedding_plot(self, batch, spatial1, uu_edges=None, ui_edges=None, ii_edges=None, spatial2=None, label="embeddings"):

        # It's expensive to make this plot with too many tracks!
        if self.hparams["num_tracks"] <= 5 and self.logger.experiment is not None:
            self.logger.experiment.log(
                {
                label: self.plot_embedding_native(batch, spatial1, spatial2, uu_edges, ui_edges, ii_edges),
                "original_space": self.plot_original_space(batch, spatial1, spatial2, ui_edges),
                }
            )

    def plot_original_space(self, batch, spatial1, spatial2=None, ui_edges=None):

        fig = go.Figure()

        if ui_edges.shape[1] > 0:

            users = ui_edges[0].unique().cpu()
            representatives = ui_edges[1].unique().cpu()

            user_embed = spatial1[users]
            influencer_embed = spatial2[representatives]
            all_embed = torch.cat([user_embed, influencer_embed], dim=0)

            # Use an incremental PCA every 10 epochs
            if self.original_pca is None:
                self.original_pca = IncrementalPCA(n_components=1)
                self.original_pca.partial_fit(all_embed.cpu().numpy())
            elif self.trainer.current_epoch % 10 == 0:
                self.original_pca.partial_fit(all_embed.cpu().numpy())

            # Transform the embeddings
            all_pca = self.original_pca.transform(all_embed.cpu().numpy())
            user_pca = self.original_pca.transform(user_embed.cpu().numpy())
            influencer_pca = self.original_pca.transform(influencer_embed.cpu().numpy())

            
            # plot the user embedding, with color given by the position in 1D PCA
            # Make a color scale from all_pca
            fig.add_trace(go.Scatter(x=batch.x[users,0].cpu(), y=batch.x[users,1].cpu(), mode='markers', marker=dict(cmin = all_pca.min(), cmax = all_pca.max(), color=user_pca[:,0], colorscale='Rainbow')))

            # plot the representatives, with color given by the position in 1D PCA
            fig.add_trace(go.Scatter(x=batch.x[representatives,0].cpu() + 0.01, y=batch.x[representatives,1].cpu() + 0.01, mode='markers', marker=dict(cmin = all_pca.min(), cmax = all_pca.max(), color=influencer_pca[:, 0], colorscale='Rainbow', symbol="star", size=14)))

        return fig


    def plot_embedding_native(self, batch, spatial1, spatial2=None, uu_edges=None, ui_edges=None, ii_edges=None):

        fig = go.Figure()

        # Partial fit PCA to both spatial1 and spatial2
        all_embed = torch.cat([spatial1, spatial2], dim=0).cpu().numpy()
        if self.embedding_pca is None:
            self.embedding_pca = IncrementalPCA(n_components=2)
            self.embedding_pca.partial_fit(all_embed)
        elif self.trainer.current_epoch % 10 == 0:
            self.embedding_pca.partial_fit(all_embed)

        # Transform the embeddings
        spatial1_pca = self.embedding_pca.transform(spatial1.cpu().numpy())
        if spatial2 is not None:
            spatial2_pca = self.embedding_pca.transform(spatial2.cpu().numpy())

        # Plot the edges
        # if spatial2 is None:
        #     spatial1_cpu, spatial2_cpu = spatial1.cpu(), spatial1.cpu()
        # else:
        #     spatial1_cpu, spatial2_cpu = spatial1.cpu(), spatial2.cpu()
        
        if uu_edges is not None and uu_edges.shape[1] > 0:
            self.plot_edges(fig, uu_edges, spatial1_pca, spatial1_pca, batch.pid, color1="green", color2="orange")

        if ii_edges is not None and ii_edges.shape[1] > 0:
            self.plot_edges(fig, ii_edges, spatial2_pca, spatial2_pca, batch.pid, color1="purple", color2="grey")

        if ui_edges is not None and ui_edges.shape[1] > 0:
            self.plot_edges(fig, ui_edges, spatial1_pca, spatial2_pca, batch.pid, color1="blue", color2="red")

        # Create a map of PID to color
        all_pids = batch.pid.unique()
        color_list = list(range(len(all_pids)))
        pid_to_color = dict(zip(all_pids.cpu().numpy(), color_list))

        # Plot spatial1 points as circles, with each point colored, with a map of PID to color
        fig.add_trace(
            go.Scatter(
                x=spatial1_pca[:, 0],
                y=spatial1_pca[:, 1],
                mode="markers",
                marker=dict(
                    cmax=len(all_pids),
                    cmin=0,
                    size=10,
                    color=[pid_to_color[pid] for pid in batch.pid.cpu().numpy()],
                    colorscale="Rainbow"
                )
            )
        )

        if spatial2 is not None:
            # Plot spatial2 points as stars, with each point colored, with a map of PID to color
            fig.add_trace(
                go.Scatter(
                    x=spatial2_pca[:, 0],
                    y=spatial2_pca[:, 1],
                    mode="markers",
                    marker=dict(
                        size=8,
                        cmax=len(all_pids),
                        cmin=0,
                        color=[pid_to_color[pid] for pid in batch.pid.cpu().numpy()],
                        symbol="star",
                        colorscale="Rainbow"
                        )
                    )
                )
            
            if ui_edges is not None and ui_edges.shape[1] > 0:
                for pid in batch.pid.unique():
                    rep_with_pid = ui_edges[1, batch.pid[ui_edges[1]] == pid].unique().cpu()

                    spatial2_rep = spatial2_pca[rep_with_pid]
                    # Ensure that spatial2_rep is still a 2D numpy array
                    if len(spatial2_rep.shape) == 1:
                        spatial2_rep = spatial2_rep.reshape(1, -1)

                    fig.add_trace(
                        go.Scatter(
                            x=spatial2_rep[:, 0],
                            y=spatial2_rep[:, 1],
                            mode="markers",
                            marker=dict(
                                size=14,
                                cmax=len(all_pids),
                                cmin=0,
                                color=[pid_to_color[pid] for pid in batch.pid[rep_with_pid].cpu().numpy()],
                                symbol="star",
                                colorscale="Rainbow"
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


    def get_cluster_metrics(self, batch, user_user_edges, user_user_truth):
        # Compute the cluster metrics
        cluster_true = batch.edge_index.shape[1]
        cluster_true_positive = user_user_truth.sum()
        cluster_positive = user_user_edges.shape[1]

        cluster_eff = cluster_true_positive / max(cluster_true, 1)
        cluster_pur = cluster_true_positive / max(cluster_positive, 1)

        return cluster_eff, cluster_pur

    def get_representative_metrics(self, batch, user_influencer_edges, user_influencer_truth):
        # Compute the representative metrics
        representative_true_positive = user_influencer_truth.sum()
        representative_positive = user_influencer_edges.shape[1]

        representative_true = 0
        represent_dup = []
        for pid, particle_length in torch.stack(batch.pid.unique(return_counts=True)).T:
            if particle_length > self.hparams["num_layer"]:
                print("Particle length is greater than num layer")
                continue
            num_reps = max(1, user_influencer_edges[1, batch.pid[user_influencer_edges[1]] == pid].unique().shape[0])
            representative_true += num_reps * particle_length
            represent_dup.append(num_reps - 1)

        represent_eff = representative_true_positive / max(representative_true, 1)
        represent_pur = representative_true_positive / max(representative_positive, 1)
        represent_dup = torch.tensor(represent_dup).float().mean()

        return represent_eff, represent_pur, represent_dup

    def get_tracking_metrics(self, batch, user_influencer_edges, user_influencer_truth):
        """
        We calculate tracking efficiency, tracking fake rate and tracking duplicity.
        We loop through each (valid) PID and check how many representative (valid) clusters match to that track.
        A cluster X is matched to PID Y if strictly greater than 50% of the hits in the cluster X belong to PID Y, and vice-versa. 
        A cluster/PID is valid if it has at least 3 hits.
        The tracking efficiency is the number of valid PIDs matched to a cluster divided by the number of valid PIDs.
        The tracking fake rate is the number of valid clusters not matched to any PID divided by the number of valid clusters.
        The tracking duplicity is the number of valid clusters matched to more than one PID divided by the number of valid clusters.
        """

        clusters = pd.DataFrame(user_influencer_edges.cpu().T.numpy(), columns=['hits', 'cluster_id'])
        clusters['cluster_size'] = clusters.groupby('cluster_id')['cluster_id'].transform('count')
        clusters['cluster_valid'] = clusters['cluster_size'] >= 3
        # make a particles df where the index is the hits id
        particles = pd.DataFrame(batch.pid.cpu().numpy(), columns=['pid'])
        particles = particles.reset_index().rename(columns={'index': 'hits'})
        # add a count of each pid
        particles['pid_size'] = particles.groupby('pid')['pid'].transform('count')
        particles['pid_valid'] = particles['pid_size'] >= 3

        clusters = clusters.merge(particles, on='hits').rename(columns={'pid': 'hit_pid'})
        clusters = clusters.merge(particles[["pid", "hits"]], left_on='cluster_id', right_on='hits', suffixes=('', '_cluster')).drop(columns=['hits_cluster']).rename(columns={'pid': 'cluster_pid'})
        clusters["matching"] = clusters["hit_pid"] == clusters["cluster_pid"]

        # Get number of matching for each cluster_id
        clusters["cluster_matching_count"] = clusters.groupby("cluster_id")["matching"].transform("sum")
        clusters["matched"] = (clusters["cluster_matching_count"] > clusters["cluster_size"]/2) & (clusters["cluster_matching_count"] > clusters["pid_size"]/2) & (clusters["cluster_valid"]) & (clusters["pid_valid"])

        # Get number of particles that are matched by at least one cluster
        particles["matched"] = particles["pid"].isin(clusters[clusters["matched"]]["cluster_pid"].unique())
        num_valid_particles = particles[particles["pid_valid"]]["pid"].nunique()
        num_valid_matched_particles = particles[particles["matched"]]["pid"].nunique()

        # Get number of clusters that are matched to any particles
        num_valid_clusters = clusters[clusters["cluster_valid"]]["cluster_id"].nunique()
        num_valid_matched_clusters = clusters[clusters["matched"]]["cluster_id"].nunique()

        # Get duplicated matched clusters by count number of unique cluster IDs for each cluster PID
        num_duplicated_clusters = (clusters[clusters["matched"]].groupby(["cluster_pid"])["cluster_id"].nunique() - 1).sum()

        tracking_eff = num_valid_matched_particles / max(num_valid_particles, 1)
        tracking_pur = num_valid_matched_clusters / max(num_valid_clusters, 1)
        tracking_dup = num_duplicated_clusters / max(num_valid_matched_clusters, 1)

        print(f"Tracking eff: {tracking_eff}, tracking pur: {tracking_pur}, tracking dup: {tracking_dup}, \n num_valid_matched_particles: {num_valid_matched_particles}, num_valid_particles: {num_valid_particles}, num_valid_matched_clusters: {num_valid_matched_clusters}, num_valid_clusters: {num_valid_clusters}, num_duplicated_clusters: {num_duplicated_clusters}")

        return tracking_eff, tracking_pur, tracking_dup