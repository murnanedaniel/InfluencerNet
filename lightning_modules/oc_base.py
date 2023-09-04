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
from .influencer_base import InfluencerBase
from .utils import build_edges
from torch_geometric.data import Dataset, Batch
import copy

from .generation_utils import graph_intersection, generate_toy_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
sqrt_eps = 1e-12

class OCBase(InfluencerBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """
        self.save_hyperparameters(hparams)

    def get_training_edges(self, batch, embedding_query, embedding_database, hnm=False, rp=False, tp=False, radius=None, knn=None, batch_index=None):

        # print(batch_index.max())

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

    def training_step(self, batch, batch_idx):

        # print(batch.batch)

        """
        The OC training step. 
        1. Runs the model in no_grad mode to get the user and influencer embeddings
        2. Builds hard negative, random pairs, and true edges for the user-user loss
        3. Calculate the potential V of each track
        4. Calculate attractive and repulsive V losses
        5. Calculate the single-condensation-point beta loss
        6. Weighted sum of losses
        """

        # Get the user and influencer embeddings
        input_data = self.get_input_data(batch)
        user_embed, beta = self(input_data, batch.batch)

        # Charge q is the arctanh2 of beta + hyperparameter qmin
        q = torch.atanh(beta)**2 + self.hparams["qmin"]

        # Calculate each loss function
        potential_loss = self.get_potential_loss(batch, user_embed, q)
        beta_loss = self.beta_weight * self.get_beta_loss(batch, beta)

        # Calculate the total loss
        loss = potential_loss + beta_loss

        self.log_dict({"train_loss": loss, "potential_loss": potential_loss, "beta_loss": beta_loss}, on_epoch=True, on_step=False)
        
        if torch.isnan(loss):
            print("Loss is nan")
            sys.exit()

        return loss

    def get_potential_loss(self, batch, user_embed, q):

        # Vpos is attractive, = |x - x_a|^2 * q_ak, where x and x_a belong to the same track
        V_pos = []
        for k in batch.pid.unique():
            # For track k, q_ak is the max q for all hits belonging to that track
            q_ak_index = q[batch.pid == k].argmax()
            q_ak = q[q_ak_index]
            x_ak = user_embed[q_ak_index]
            # print(f"batch: {batch}, user_embed.shape: {user_embed.shape}, k: {k}, q: {q}, q_ak_index.shape: {q_ak_index.shape}, q_ak.shape: {q_ak.shape}, x_ak.shape: {x_ak.shape}")
            V_ak = torch.mean(q[batch.pid == k] * torch.sum((user_embed[batch.pid == k] - x_ak)**2, dim=-1)) * q_ak
            V_pos.append(V_ak)
        
        V_pos = torch.stack(V_pos).mean() if V_pos else torch.tensor(0.0, device=self.device)
        print(f"V_pos: {V_pos}")
        # Vneg is repulsive, = max(0, 1 - |x - x_a|)*q_ak, where x and x_a belong to different tracks

        # First get all negative edges:
        all_edges = build_edges(user_embed, user_embed, r_max=1.0, k_max=100, batch_index = batch.batch)
        negative_edges = all_edges[:, batch.pid[all_edges[0]] != batch.pid[all_edges[1]]]
        V_neg = []
        for k in batch.pid.unique():
            q_ak_index = q[batch.pid == k].argmax()
            q_ak = q[q_ak_index]
            x_ak = user_embed[q_ak_index]
            q_ak_edges = negative_edges[:, negative_edges[1] == q_ak_index]
            if q_ak_edges.shape[1] == 0:
                continue
            V_ak = torch.mean(q[q_ak_edges[0]] * torch.max(torch.zeros_like(q[q_ak_edges[0]]), 1 - torch.sqrt(torch.sum((user_embed[q_ak_edges[0]] - x_ak)**2, dim=-1) + sqrt_eps))) * q_ak
            if torch.isnan(V_ak):
                print(f"V_ak: {V_ak}, q_ak: {q_ak}, q_ak_index: {q_ak_index}, q: {q}, q_ak_edges: {q_ak_edges}, user_embed[q_ak_edges[0]]: {user_embed[q_ak_edges[0]]}, x_ak: {x_ak}")
            V_neg.append(V_ak)

        V_neg = torch.stack(V_neg).mean() if V_neg else torch.tensor(0.0, device=self.device)
        print(f"V_neg: {V_neg}")
        # L is the sum of Vpos and Vneg
        return V_pos + V_neg

    def get_beta_loss(self, batch, beta):

        beta_loss = []
        for k in batch.pid.unique():
            beta_ak = beta[batch.pid == k].max()
            beta_loss.append(1 - beta_ak)

        beta_loss = torch.stack(beta_loss).mean()
        print(beta_loss)

        return beta_loss

    def get_condensation_edges(self, batch, user_embed, beta):

        # Trim to vertices with beta above hparam tbeta
        mask = (beta > self.hparams["tbeta"])
        print("Mask sum:", mask.sum())
        print(f"Beta distribution: {torch.mean(beta)}, {torch.std(beta)})")
        masked_beta = beta[mask]
        mask_indices = torch.where(mask)[0]

        # Build candidate edges as radius graph with beta vertices as database and all vertices as query
        candidate_edges = build_edges(user_embed, user_embed, r_max=self.hparams["td"], k_max=100, batch_index = batch.batch)

        # Sort beta vertices in descending order in beta
        _, sorted_indices = masked_beta.sort(descending=True)
        beta_indices = mask_indices[sorted_indices]
        beta = beta[beta_indices]

        print(f"Masked beta: {beta}, beta_indices: {beta_indices}")
        # Loop over vertices, and add edges to final list if vertex is not already in an edge
        condensation_edges = torch.empty([2, 0], dtype=torch.int64, device=self.device)
        for beta_index in beta_indices:
            # If the vertex is already in an edge, skip it
            if beta_index in condensation_edges:
                continue
            # Get the edges for the vertex
            vertex_edges = candidate_edges[:, candidate_edges[1] == beta_index]
            # Add the edges to the final list
            condensation_edges = torch.cat([condensation_edges, vertex_edges], dim=1)

        # Get the truth values for the edges
        condensation_truth = self.get_truth(condensation_edges, batch)

        return condensation_edges, condensation_truth

    def shared_evaluation(self, batch, batch_idx):

        input_data = self.get_input_data(batch)
        user_embed, beta = self(input_data, batch.batch)

        # Charge q is the arctanh2 of beta + hyperparameter qmin
        q = torch.atanh(beta)**2 + self.hparams["qmin"]
        
        # Calculate each loss function
        potential_loss = self.get_potential_loss(batch, user_embed, q)
        beta_loss = self.get_beta_loss(batch, beta)

        # Calculate the total loss
        loss = potential_loss + self.beta_weight * beta_loss

        # Get the condensation edges
        user_influencer_edges, user_influencer_truth = self.get_condensation_edges(batch, user_embed, beta)

        current_lr = self.optimizers().param_groups[0]["lr"]

        represent_eff, represent_pur, represent_dup = self.get_representative_metrics(batch, user_influencer_edges, user_influencer_truth)
        tracking_eff, tracking_pur, tracking_dup = self.get_tracking_metrics(batch, user_influencer_edges, user_influencer_truth)

        with contextlib.suppress(Exception):
            self.log_dict(
                {
                    "val_loss": loss,
                    "represent_pur": represent_pur,
                    "represent_eff": represent_eff,
                    "represent_dup": represent_dup,
                    "lr": current_lr,
                    "tracking_eff": tracking_eff, "tracking_fake_rate": 1-tracking_pur, "tracking_dup": tracking_dup
                },
            )
        if batch_idx == 0:
            print(f"Rep eff: {represent_eff}, rep pur: {represent_pur}, rep dup: {represent_dup}")

            first_event = Batch.to_data_list(batch)[0]
            pid_mask = torch.isin(batch.pid, first_event.pid)

            self.log_embedding_plot(batch, user_embed[pid_mask], spatial2=user_embed[pid_mask], ui_edges=user_influencer_edges[:, pid_mask[user_influencer_edges].all(dim=0)])

        return {"val_loss": loss, "represent_pur": represent_pur, "represent_eff": represent_eff, "represent_dup": represent_dup}

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
    def beta_weight(self):
        return self.get_weight("beta_weight")