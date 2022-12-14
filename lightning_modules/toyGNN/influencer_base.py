"""The base classes for the embedding process.

The embedding process here is both the embedding models (contained in Models/) and the training procedure, which is a Siamese network strategy. Here, the network is run on all points, and then pairs (or triplets) are formed by one of several strategies (e.g. random pairs (rp), hard negative mining (hnm)) upon which some sort of contrastive loss is applied. The default here is a hinge margin loss, but other loss functions can work, including cross entropy-style losses. Also available are triplet loss approaches.

Example:
    See Quickstart for a concrete example of this process.
    
Todo:
    * Refactor the training & validation pipeline, since the use of the different regimes (rp, hnm, etc.) looks very messy
"""

# System imports
import sys

import torch

# Local Imports
from .embedding_base import EmbeddingBase
from .utils import build_edges

device = "cuda" if torch.cuda.is_available() else "cpu"

class InfluencerBase(EmbeddingBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """
        self.save_hyperparameters(hparams)
        self.trainset, self.valset, self.testset = None, None, None

    def get_training_edges(self, batch, embedding_query, embedding_database, hnm=False, rp=False, tp=False, radius=None, knn=None):

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
            training_edges = self.append_hnm_pairs(training_edges, embedding_query, embedding_database, radius=radius, knn=knn)

        # Append Random Pairs (rp) with KNN graph
        if rp:
            training_edges = self.append_random_pairs(training_edges, embedding_database)

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
                margin=self.hparams["margin"]**2,
                reduction="mean",
            )
        else:
            negative_loss = 0

        if (hinge == 1).any():
            positive_loss = torch.nn.functional.hinge_embedding_loss(
                d[hinge == 1],
                hinge[hinge == 1],
                margin=self.hparams["margin"]**2,
                reduction="mean",
            )
        else:
            positive_loss = 0

        return negative_loss + self.hparams["user_user_weight"] * positive_loss

    def get_user_influencer_loss(self, user_influencer_edges, user_influencer_truth, user_embed, influencer_embed, batch):

        hinge, d = self.get_hinge_distance(user_embed, influencer_embed, user_influencer_edges, user_influencer_truth)
        positive_loss = []

        for pid, particle_length in torch.stack(batch.pid.unique(return_counts=True)).T:
            true_edges = (batch.pid[user_influencer_edges] == pid).all(dim=0)
            num_true_edges = true_edges.sum()
            if num_true_edges > 0:
                grav_weights = 1 - torch.exp( -d[true_edges] / self.hparams["margin"]**2 / self.influencer_spread)
                prod_grav_weights = grav_weights.prod()
                # grav_weights = self.influencer_spread * self.hparams["margin"]**2 / (-d[true_edges])
                # prod_grav_weights = torch.exp(grav_weights.sum())
                if self.hparams["influencer_exponent"] is not None:
                    power_grav_weights = (prod_grav_weights+1e-20).pow(1/self.hparams["influencer_exponent"])    
                else:
                    power_grav_weights = (prod_grav_weights+1e-20).pow(1/particle_length)
                positive_loss.append(power_grav_weights)
                # Check if nan
                if torch.isnan(power_grav_weights):
                    print(f"Class {pid} num edges: {num_true_edges}")
                    print(f"Class {pid} particles length: {particle_length}")
                    print(f"Class {pid} grav weights: {grav_weights} \n")
                    print(f"Class {pid} prod grav weights: {prod_grav_weights}")
                    print(f"Class {pid} loss: {power_grav_weights} \n")
                    sys.exit()

        positive_loss = torch.stack(positive_loss).mean()
        negative_loss = torch.stack([self.hparams["margin"]**2 - d[hinge==-1], torch.zeros_like(d[hinge==-1])], dim=1).max(dim=1)[0].mean()
        
        loss = 0
        if not torch.isnan(negative_loss):
            loss += negative_loss
        if not torch.isnan(positive_loss):
            loss += self.hparams["user_influencer_weight"] * positive_loss
        return loss
 
    def get_influencer_influencer_loss(self, influencer_influencer_edges, influencer_influencer_truth, influencer_embed):

        _, d = self.get_hinge_distance(influencer_embed, influencer_embed, influencer_influencer_edges, influencer_influencer_truth)

        return self.hparams["influencer_influencer_weight"] * torch.stack([(2*self.hparams["margin"])**2 - d, torch.zeros_like(d)], dim=1).max(dim=1)[0].mean()
        
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
            user_embed, influencer_embed = self(input_data)

        # Get the training edges for each loss function
        user_user_edges, user_user_truth = self.get_training_edges(batch, user_embed, user_embed, hnm=True, rp=True, tp=True)
        user_influencer_edges, user_influencer_truth = self.get_training_edges(batch, user_embed, influencer_embed, hnm=True, tp=True, rp=True)
        influencer_influencer_edges, influencer_influencer_truth = self.get_training_edges(batch, influencer_embed, influencer_embed, hnm=True, rp=True, radius=2*self.hparams["radius"])

        # Get the hits of interest
        included_hits = torch.cat([user_user_edges, user_influencer_edges, influencer_influencer_edges], dim=1).unique()
        user_embed[included_hits], influencer_embed[included_hits] = self(input_data[included_hits])
        
        # Calculate each loss function
        user_user_loss = self.get_user_user_loss(user_user_edges, user_user_truth, user_embed)
        user_influencer_loss = self.get_user_influencer_loss(user_influencer_edges, user_influencer_truth, user_embed, influencer_embed, batch)
        influencer_influencer_loss = self.get_influencer_influencer_loss(influencer_influencer_edges, influencer_influencer_truth, influencer_embed)

        # Compute the total loss
        if self.user_influencer_loss_ratio is not None:
            loss = self.user_influencer_loss_ratio[0]*user_user_loss + self.user_influencer_loss_ratio[1]*user_influencer_loss + influencer_influencer_loss
        elif self.user_user_loss_ratio is not None:
            loss = self.user_user_loss_ratio*user_user_loss + user_influencer_loss + self.user_user_loss_ratio*influencer_influencer_loss
        else:
            loss = user_user_loss + user_influencer_loss + influencer_influencer_loss

        self.log_dict({"train_loss": loss, "train_user_user_loss": user_user_loss, "train_user_influencer_loss": user_influencer_loss, "train_influencer_influencer_loss": influencer_influencer_loss})

        if torch.isnan(loss):
            print("Loss is nan")
            print(user_embed, influencer_embed)
            print(user_user_loss, user_influencer_loss, influencer_influencer_loss)
            print(user_user_edges, user_user_truth)
            print(user_influencer_edges, user_influencer_truth)
            print(influencer_influencer_edges, influencer_influencer_truth)

            sys.exit()

        return loss

    def shared_evaluation(self, batch, batch_idx):

        input_data = self.get_input_data(batch)
        user_embed, influencer_embed = self(input_data)

        try:
            user_user_edges, user_user_truth = self.get_training_edges(batch, user_embed, user_embed, hnm=True, knn=500)
            user_user_loss = self.get_user_user_loss(user_user_edges, user_user_truth, user_embed)
        except Exception:
            user_user_edges, user_user_truth = torch.empty([2, 0], dtype=torch.int64, device=self.device), torch.empty([0], dtype=torch.int64, device=self.device)
            user_user_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        try:
            user_influencer_edges, user_influencer_truth = self.get_training_edges(batch, user_embed, influencer_embed, hnm=True, knn=500)
            user_influencer_loss = self.get_user_influencer_loss(user_influencer_edges, user_influencer_truth, user_embed, influencer_embed, batch)
        except Exception:
            user_influencer_edges, user_influencer_truth = torch.empty([2, 0], dtype=torch.int64, device=self.device), torch.empty([0], dtype=torch.int64, device=self.device)
            user_influencer_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        try:
            influencer_influencer_edges, influencer_influencer_truth = self.get_training_edges(batch, influencer_embed, influencer_embed, hnm=True, radius=2*self.hparams["radius"], knn=500)    
            influencer_influencer_loss = self.get_influencer_influencer_loss(influencer_influencer_edges, influencer_influencer_truth, influencer_embed)
        except Exception:
            # print("No influencer-influencer edges")
            influencer_influencer_edges, influencer_influencer_truth = torch.empty([2, 0], dtype=torch.int64, device=self.device), torch.empty([0], dtype=torch.int64, device=self.device)
            influencer_influencer_loss = torch.tensor(0, dtype=torch.float32, device=self.device)

        # Compute the total loss
        loss = user_user_loss + user_influencer_loss + influencer_influencer_loss

        current_lr = self.optimizers().param_groups[0]["lr"]

        cluster_eff, cluster_pur = self.get_cluster_metrics(batch, user_user_edges, user_user_truth)
        represent_eff, represent_pur, represent_dup = self.get_representative_metrics(batch, user_influencer_edges, user_influencer_truth)

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
            
            self.log_embedding_plot(batch, user_embed, spatial2=influencer_embed, uu_edges=user_user_edges, ui_edges=user_influencer_edges, ii_edges=influencer_influencer_edges)

        return loss

    def get_cluster_metrics(self, batch, user_user_edges, user_user_truth):
        # Compute the cluster metrics
        cluster_true = batch.all_signal_edges.shape[1]
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
            num_reps = max(1, user_influencer_edges[1, batch.pid[user_influencer_edges[1]] == pid].unique().shape[0])
            representative_true += num_reps * particle_length
            represent_dup.append(num_reps - 1)

        represent_eff = representative_true_positive / max(representative_true, 1)
        represent_pur = representative_true_positive / max(representative_positive, 1)
        represent_dur = torch.tensor(represent_dup).float().mean()

        return represent_eff, represent_pur, represent_dur


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