"""The base classes for the embedding process.

The embedding process here is both the embedding models (contained in Models/) and the training procedure, which is a Siamese network strategy. Here, the network is run on all points, and then pairs (or triplets) are formed by one of several strategies (e.g. random pairs (rp), hard negative mining (hnm)) upon which some sort of contrastive loss is applied. The default here is a hinge margin loss, but other loss functions can work, including cross entropy-style losses. Also available are triplet loss approaches.

Example:
    See Quickstart for a concrete example of this process.
    
Todo:
    * Refactor the training & validation pipeline, since the use of the different regimes (rp, hnm, etc.) looks very messy
"""


import contextlib
# System imports
import sys

import torch

# Local Imports
from .embedding_base import EmbeddingBase
from .utils import build_edges
from torch_geometric.data import Dataset, Batch
import copy

from .generation_utils import graph_intersection, generate_toy_dataset
from .utils import load_datafiles_in_dir, handle_hard_node_cuts, map_tensor_handler

device = "cuda" if torch.cuda.is_available() else "cpu"
sqrt_eps = 1e-12

class InfluencerBase(EmbeddingBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """
        self.save_hyperparameters(hparams)
        if "dataset" not in self.hparams or self.hparams["dataset"] == "toy":
            self.dataset_class = ToyDataset
        elif self.hparams["dataset"] == "trackml":
            self.dataset_class = TrackMLDataset

        self.pca = None

    def get_training_edges(self, batch, embedding_query, embedding_database, hnm=False, rp=False, tp=False, radius=None, knn=None, batch_index=None):

        print(batch_index.max())

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
                d[hinge == 1], 
                hinge[hinge == 1],
                # margin=self.hparams["margin"]**2, # SQREUCLIDEAN
                margin=self.hparams["margin"], # EUCLIDEAN
                reduction="mean",
            )
        else:
            positive_loss = 0

        return self.user_user_weight * (negative_loss + self.hparams["user_user_pos_ratio"] * positive_loss)

    # Static method to calculate geometric mean
    @staticmethod
    def geometric_mean(x, dim=0):
        return torch.exp(torch.mean(torch.log(x), dim=dim))

    def get_user_influencer_loss(self, user_influencer_edges, user_influencer_truth, user_embed, influencer_embed, batch):
        
        positive_loss = []

        pid, particle_length = batch.pid.unique(return_counts=True)
        if particle_length.max() > self.hparams["num_layer"]:
            print("Too many particles in batch", particle_length[particle_length > self.hparams["num_layer"]], pid[particle_length > self.hparams["num_layer"]])

        for pid, particle_length in torch.stack(batch.pid.unique(return_counts=True)).T:
            if particle_length > self.hparams["num_layer"]:
                continue
            true_hits = torch.where(batch.pid == pid)[0]
            true_mesh = torch.meshgrid(true_hits, true_hits)

            dist_sq = torch.sum((user_embed[true_mesh[0]] - influencer_embed[true_mesh[1]])**2, dim=-1) # SQREUCLIDEAN
            dist = torch.sqrt(dist_sq + sqrt_eps) # EUCLIDEAN
            # follower_sum = dist_sq.mean(dim=0) / (self.hparams["margin"]**2 * self.influencer_spread**2) # SQREUCLIDEAN
            follower_sum = dist.mean(dim=0) / (self.hparams["margin"] * self.influencer_spread) # EUCLIDEAN
            influencer_prod = self.geometric_mean(follower_sum, dim=0)
            
            # Check if influencer_prod is nan or inf
            if influencer_prod != influencer_prod or influencer_prod == float(
                "inf"
            ):
                print(f"Influencer prod is nan or inf... \n Influencer prod: {influencer_prod} \n pid: {pid} \n particle_length: {particle_length} \n dist_sq: {dist_sq} \n dist: {dist} \n follower_sum: {follower_sum} \n user_embed: {user_embed[true_mesh[0]]} \n influencer_embed: {influencer_embed[true_mesh[1]]}")
                sys.exit()

            positive_loss.append(influencer_prod)

        positive_loss = torch.stack(positive_loss).mean()

        hinge, d = self.get_hinge_distance(user_embed, influencer_embed, user_influencer_edges, user_influencer_truth)
        # negative_loss = torch.stack([self.hparams["margin"]**2 - d[hinge==-1], torch.zeros_like(d[hinge==-1])], dim=1).max(dim=1)[0].mean() # SQREUCLIDEAN
        negative_loss = torch.stack([self.hparams["margin"] - d[hinge==-1], torch.zeros_like(d[hinge==-1])], dim=1).max(dim=1)[0].mean() # EUCLIDEAN

        # print("Positive loss", positive_loss, "Negative loss", negative_loss, "Weight", self.user_influencer_weight)

        loss = 0
        if not torch.isnan(negative_loss):
            loss += negative_loss
        if not torch.isnan(positive_loss):
            loss += self.hparams["user_influencer_pos_ratio"] * positive_loss
        return self.user_influencer_weight * loss
 
    def get_influencer_influencer_loss(self, influencer_influencer_edges, influencer_influencer_truth, influencer_embed):

        _, d = self.get_hinge_distance(influencer_embed, influencer_embed, influencer_influencer_edges, influencer_influencer_truth)

        # return self.influencer_influencer_weight * torch.stack([(self.hparams["influencer_margin"])**2 - d, torch.zeros_like(d)], dim=1).max(dim=1)[0].mean() # SQREUCLIDEAN
        return self.influencer_influencer_weight * torch.stack([(self.hparams["influencer_margin"]) - d, torch.zeros_like(d)], dim=1).max(dim=1)[0].mean() # EUCLIDEAN
        
    def training_step(self, batch, batch_idx):

        print(batch.batch)

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
            user_embed, influencer_embed = self(input_data, batch.batch)

        print(batch.batch)
        print(batch.batch.max())

        # Get the training edges for each loss function
        user_user_edges, user_user_truth = self.get_training_edges(batch, user_embed, user_embed, hnm=True, rp=True, tp=True, batch_index = batch.batch)
        user_influencer_edges, user_influencer_truth = self.get_training_edges(batch, user_embed, influencer_embed, hnm=True, tp=True, rp=True, batch_index = batch.batch)
        influencer_influencer_edges, influencer_influencer_truth = self.get_training_edges(batch, influencer_embed, influencer_embed, hnm=True, rp=True, radius=self.hparams["influencer_margin"], batch_index = batch.batch)

        # Get the hits of interest
        included_hits = torch.cat([user_user_edges, user_influencer_edges, influencer_influencer_edges], dim=1).unique()
        user_embed[included_hits], influencer_embed[included_hits] = self(input_data[included_hits], batch.batch[included_hits])
        
        # Calculate each loss function
        user_user_loss = self.get_user_user_loss(user_user_edges, user_user_truth, user_embed)
        user_influencer_loss = self.get_user_influencer_loss(user_influencer_edges, user_influencer_truth, user_embed, influencer_embed, batch)
        influencer_influencer_loss = self.get_influencer_influencer_loss(influencer_influencer_edges, influencer_influencer_truth, influencer_embed)

        loss = user_user_loss + user_influencer_loss + influencer_influencer_loss

        self.log_dict({"train_loss": loss, "train_user_user_loss": user_user_loss, "train_user_influencer_loss": user_influencer_loss, "train_influencer_influencer_loss": influencer_influencer_loss})
        
        # print(f"User user loss: {user_user_loss} \n User influencer loss: {user_influencer_loss} \n Influencer influencer loss: {influencer_influencer_loss} \n Total loss: {loss}")
        # print(f"User user edges: {user_user_edges} \n User influencer edges: {user_influencer_edges} \n Influencer influencer edges: {influencer_influencer_edges}")
        # print(f"User user truth: {user_user_truth} \n User influencer truth: {user_influencer_truth} \n Influencer influencer truth: {influencer_influencer_truth}")

        if torch.isnan(loss):
            print("Loss is nan")
            sys.exit()

        return loss

    def shared_evaluation(self, batch, batch_idx):

        print(batch.batch)

        input_data = self.get_input_data(batch)
        user_embed, influencer_embed = self(input_data, batch.batch)

        print(batch.batch)

        try:
            user_user_edges, user_user_truth = self.get_training_edges(batch, user_embed, user_embed, hnm=True, knn=500, batch_index=batch.batch)
            user_user_loss = self.get_user_user_loss(user_user_edges, user_user_truth, user_embed)
        except Exception:
            user_user_edges, user_user_truth = torch.empty([2, 0], dtype=torch.int64, device=self.device), torch.empty([0], dtype=torch.int64, device=self.device)
            user_user_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        try:
            user_influencer_edges, user_influencer_truth = self.get_training_edges(batch, user_embed, influencer_embed, hnm=True, knn=500, batch_index=batch.batch)
            user_influencer_loss = self.get_user_influencer_loss(user_influencer_edges, user_influencer_truth, user_embed, influencer_embed, batch)
        except Exception:
            user_influencer_edges, user_influencer_truth = torch.empty([2, 0], dtype=torch.int64, device=self.device), torch.empty([0], dtype=torch.int64, device=self.device)
            user_influencer_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        try:
            influencer_influencer_edges, influencer_influencer_truth = self.get_training_edges(batch, influencer_embed, influencer_embed, hnm=True, radius=self.hparams["influencer_margin"], knn=500, batch_index=batch.batch)  
            influencer_influencer_loss = self.get_influencer_influencer_loss(influencer_influencer_edges, influencer_influencer_truth, influencer_embed)
        except Exception:
            # print("No influencer-influencer edges")
            influencer_influencer_edges, influencer_influencer_truth = torch.empty([2, 0], dtype=torch.int64, device=self.device), torch.empty([0], dtype=torch.int64, device=self.device)
            influencer_influencer_loss = torch.tensor(0, dtype=torch.float32, device=self.device)

        # Compute the total loss
        loss = user_user_loss + user_influencer_loss + influencer_influencer_loss
        # loss = user_influencer_loss + influencer_influencer_loss

        current_lr = self.optimizers().param_groups[0]["lr"]

        cluster_eff, cluster_pur = self.get_cluster_metrics(batch, user_user_edges, user_user_truth)
        represent_eff, represent_pur, represent_dup = self.get_representative_metrics(batch, user_influencer_edges, user_influencer_truth)

        with contextlib.suppress(Exception):
            self.log_dict(
                {
                    "val_loss": loss, "cluster_eff": cluster_eff, "cluster_pur": cluster_pur, "represent_pur": represent_pur, "represent_eff": represent_eff, "represent_dup": represent_dup,
                    "lr": current_lr, "user_user_loss": user_user_loss, "user_influencer_loss": user_influencer_loss, "influencer_influencer_loss": influencer_influencer_loss
                },
            )
        if batch_idx == 0:
            print(f"Influencer spread: {self.influencer_spread}")
            print(f"Rep eff: {represent_eff}, rep pur: {represent_pur}, rep dup: {represent_dup}")
            print(f"Cluster eff: {cluster_eff}, cluster pur: {cluster_pur}")

            first_event = Batch.to_data_list(batch)[0]
            pid_mask = torch.isin(batch.pid, first_event.pid)

            # self.log_embedding_plot(batch, user_embed[pid_mask], spatial2=influencer_embed[pid_mask], uu_edges=user_user_edges, ui_edges=user_influencer_edges, ii_edges=influencer_influencer_edges)
            self.log_embedding_plot(batch, user_embed[pid_mask], spatial2=influencer_embed[pid_mask], uu_edges=user_user_edges[:, pid_mask[user_user_edges].all(dim=0)], ui_edges=user_influencer_edges[:, pid_mask[user_influencer_edges].all(dim=0)], ii_edges=influencer_influencer_edges[:, pid_mask[influencer_influencer_edges].all(dim=0)])

        return {"val_loss": loss, "cluster_eff": cluster_eff, "cluster_pur": cluster_pur, "represent_pur": represent_pur, "represent_eff": represent_eff, "represent_dup": represent_dup, "lr": current_lr,
                "user_user_loss": user_user_loss, "user_influencer_loss": user_influencer_loss, "influencer_influencer_loss": influencer_influencer_loss,
                "user_user_edges": user_user_edges, "user_user_truth": user_user_truth, "user_influencer_edges": user_influencer_edges, "user_influencer_truth": user_influencer_truth,
                "influencer_influencer_edges": influencer_influencer_edges, "influencer_influencer_truth": influencer_influencer_truth,
                "user_embed": user_embed, "influencer_embed": influencer_embed}

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
                continue
            num_reps = max(1, user_influencer_edges[1, batch.pid[user_influencer_edges[1]] == pid].unique().shape[0])
            representative_true += num_reps * particle_length
            represent_dup.append(num_reps - 1)

        represent_eff = representative_true_positive / max(representative_true, 1)
        represent_pur = representative_true_positive / max(representative_positive, 1)
        represent_dup = torch.tensor(represent_dup).float().mean()

        return represent_eff, represent_pur, represent_dup


    @property
    def influencer_spread(self):
        if not isinstance(self.hparams["influencer_spread"], list):
            return self.hparams["influencer_spread"]
        if len(self.hparams["influencer_spread"]) == 2:
            start = self.hparams["influencer_spread"][0]
            end = self.hparams["influencer_spread"][1]
            return (start * end * self.hparams["max_epochs"] / (start - end)) / (self.current_epoch + ( (end*self.hparams["max_epochs"])/(start-end) ))
        elif len(self.hparams["influencer_spread"]) == 3:
            start = self.hparams["influencer_spread"][0]
            mid = self.hparams["influencer_spread"][1]
            end = self.hparams["influencer_spread"][2]
            return (start * mid * self.hparams["max_epochs"] / 2 / (start - mid)) / (self.current_epoch + ( (mid*self.hparams["max_epochs"] / 2)/(start-mid) )) if self.current_epoch < self.hparams["max_epochs"] / 2 else (mid * end * self.hparams["max_epochs"] / 2 / (mid - end)) / (self.current_epoch - self.hparams["max_epochs"] / 2 + ( (end*self.hparams["max_epochs"] / 2)/(mid-end) ))

    # @property
    # def user_influencer_weight(self):
    #     if not isinstance(self.hparams["user_influencer_weight"], list):
    #         return self.hparams["user_influencer_weight"]
    #     if len(self.hparams["user_influencer_weight"]) == 2:
    #         start = self.hparams["user_influencer_weight"][0]
    #         end = self.hparams["user_influencer_weight"][1]
    #         return start + (end - start) * self.current_epoch / self.hparams["max_epochs"]
    #     elif len(self.hparams["user_influencer_weight"]) == 3:
    #         start = self.hparams["user_influencer_weight"][0]
    #         mid = self.hparams["user_influencer_weight"][1]
    #         end = self.hparams["user_influencer_weight"][2]
    #         return start + (mid - start) * self.current_epoch / (self.hparams["max_epochs"]/2) if self.current_epoch < self.hparams["max_epochs"] / 2 else mid + (end - mid) * (self.current_epoch - self.hparams["max_epochs"] / 2) / (self.hparams["max_epochs"]/2)
    
    # @property
    # def influencer_influencer_weight(self):
    #     if not isinstance(self.hparams["influencer_influencer_weight"], list):
    #         return self.hparams["influencer_influencer_weight"]
    #     if len(self.hparams["influencer_influencer_weight"]) == 2:
    #         start = self.hparams["influencer_influencer_weight"][0]
    #         end = self.hparams["influencer_influencer_weight"][1]
    #         return start + (end - start) * self.current_epoch / self.hparams["max_epochs"]
    #     elif len(self.hparams["influencer_influencer_weight"]) == 3:
    #         start = self.hparams["influencer_influencer_weight"][0]
    #         mid = self.hparams["influencer_influencer_weight"][1]
    #         end = self.hparams["influencer_influencer_weight"][2]
    #         return start + (mid - start) * self.current_epoch / (self.hparams["max_epochs"]/2) if self.current_epoch < self.hparams["max_epochs"] / 2 else mid + (end - mid) * (self.current_epoch - self.hparams["max_epochs"] / 2) / (self.hparams["max_epochs"]/2)
    
    # @property
    # def user_user_weight(self):
    #     if not isinstance(self.hparams["user_user_weight"], list):
    #         return self.hparams["user_user_weight"]
    #     if len(self.hparams["user_user_weight"]) == 2:
    #         start = self.hparams["user_user_weight"][0]
    #         end = self.hparams["user_user_weight"][1]
    #         return start + (end - start) * self.current_epoch / self.hparams["max_epochs"]
    #     elif len(self.hparams["user_user_weight"]) == 3:
    #         start = self.hparams["user_user_weight"][0]
    #         mid = self.hparams["user_user_weight"][1]
    #         end = self.hparams["user_user_weight"][2]
    #         return start + (mid - start) * self.current_epoch / (self.hparams["max_epochs"]/2) if self.current_epoch < self.hparams["max_epochs"] / 2 else mid + (end - mid) * (self.current_epoch - self.hparams["max_epochs"] / 2) / (self.hparams["max_epochs"]/2)

    # Abstract the above into a function
    def get_weight(self, weight_name):
        if not isinstance(self.hparams[weight_name], list):
            return self.hparams[weight_name]
        if len(self.hparams[weight_name]) == 2:
            start = self.hparams[weight_name][0]
            end = self.hparams[weight_name][1]
            return start + (end - start) * self.current_epoch / self.hparams["max_epochs"]
        elif len(self.hparams[weight_name]) == 3:
            start = self.hparams[weight_name][0]
            mid = self.hparams[weight_name][1]
            end = self.hparams[weight_name][2]
            return start + (mid - start) * self.current_epoch / (self.hparams["max_epochs"]/2) if self.current_epoch < self.hparams["max_epochs"] / 2 else mid + (end - mid) * (self.current_epoch - self.hparams["max_epochs"] / 2) / (self.hparams["max_epochs"]/2)
        
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
    def user_influencer_loss_ratio(self):
        if "user_influencer_loss_ratio" not in self.hparams or self.hparams["user_influencer_loss_ratio"] is False:
            return None
        else:
            return (1 - (self.current_epoch / self.hparams["max_epochs"]), self.current_epoch / self.hparams["max_epochs"])
        
    @property
    def user_user_loss_ratio(self):
        if "user_user_loss_ratio" not in self.hparams or self.hparams["user_user_loss_ratio"] is False:
            return None
        else:
            return self.current_epoch / self.hparams["max_epochs"]

class ToyDataset(Dataset):

    def __init__(self, num_events=None, hparams=None, input_dir=None, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(transform, pre_transform, pre_filter)
        self.hparams = hparams

        # def generate_toy_dataset(num_events, num_tracks, track_dis_width, num_layers, min_r, max_r, detector_width, ptcut, nhits):

        self.events = generate_toy_dataset(num_events, hparams["num_tracks"], hparams["track_dis_width"], hparams["num_layer"], hparams["min_r"], hparams["max_r"], hparams["detector_width"], hparams["ptcut"], hparams["nhits"])

        print(f"Generated {len(self.events)} events")

    def __len__(self):
        return len(self.events)
    
    def __getitem__(self, idx):
        return self.events[idx]
    
    def collate(self, data_list):
        batch = Batch.from_data_list(data_list)


class TrackMLDataset(Dataset):
    """
    The custom default GNN dataset to load graphs off the disk
    """

    def __init__(self, input_dir=None, data_name = None, num_events = None, stage="fit", hparams=None, transform=None, pre_transform=None, pre_filter=None, **kwargs):
        super().__init__(input_dir, transform, pre_transform, pre_filter)
        
        self.input_dir = input_dir
        self.data_name = data_name
        self.hparams = hparams
        self.num_events = num_events
        self.stage = stage
        
        self.input_paths = load_datafiles_in_dir(self.input_dir, self.data_name, self.num_events)
        self.input_paths.sort() # We sort here for reproducibility
        
    def len(self):
        return len(self.input_paths)

    def get(self, idx):

        event_path = self.input_paths[idx]
        event = torch.load(event_path, map_location=torch.device("cpu"))
        self.preprocess_event(event)

        return event

    def preprocess_event(self, event):
        """
        Process event before it is used in training and validation loops
        """
        
        self.cleaning_and_tests(event)
        self.apply_hard_cuts(event)
        # self.remove_split_cluster_truth(event) TODO: Should handle this at some point
        self.scale_features(event)
        self.make_truth(event)
        
    def apply_hard_cuts(self, event):
        """
        Apply hard cuts to the event. This is implemented by 
        1. Finding which true edges are from tracks that pass the hard cut.
        2. Pruning the input graph to only include nodes that are connected to these edges.
        """
        
        if self.hparams is not None and "hard_cuts" in self.hparams.keys() and self.hparams["hard_cuts"]:
            assert isinstance(self.hparams["hard_cuts"], dict), "Hard cuts must be a dictionary"
            handle_hard_node_cuts(event, self.hparams["hard_cuts"])
            
    def cleaning_and_tests(self, event):
        """
        Ensure that data is clean and has the correct shape
        """

        if not hasattr(event, "num_nodes"):
            assert "x" in event.keys, "No node features found in event"
            event.num_nodes = event.x.shape[0]

    def scale_features(self, event):
        """
        Handle feature scaling for the event
        """
        
        if self.hparams is not None and "node_scales" in self.hparams.keys() and "node_features" in self.hparams.keys():
            assert isinstance(self.hparams["node_scales"], list), "Feature scaling must be a list of ints or floats"
            for i, feature in enumerate(self.hparams["node_features"]):
                assert feature in event.keys, f"Feature {feature} not found in event"
                event[feature] = event[feature] / self.hparams["node_scales"][i]

    def unscale_features(self, event):
        """
        Unscale features when doing prediction
        """
        
        if self.hparams is not None and "node_scales" in self.hparams.keys() and "node_features" in self.hparams.keys():
            assert isinstance(self.hparams["node_scales"], list), "Feature scaling must be a list of ints or floats"
            for i, feature in enumerate(self.hparams["node_features"]):
                assert feature in event.keys, f"Feature {feature} not found in event"
                event[feature] = event[feature] * self.hparams["node_scales"][i]

    def make_truth(self, event):
        """
        Make truth edges and truth nodes for the event
        """
        
        node_pids = map_tensor_handler(event.particle_id, "node-like", track_edges = event.track_edges, num_nodes = event.num_nodes)
        edge_index = []
        for pid in torch.unique(node_pids):
            node_index = torch.where(node_pids == pid)[0]
            # Make a meshgrid of all the nodes
            meshgrid = torch.meshgrid(node_index, node_index)
            # Flatten the meshgrid
            edge_index.append(torch.stack(meshgrid).flatten(1))

        event.edge_index = torch.cat(edge_index, dim = 1)
        event.x = torch.stack([event.r, event.phi, event.z], axis=-1).float()
        event.pid = event.particle_id

        print(event.batch)