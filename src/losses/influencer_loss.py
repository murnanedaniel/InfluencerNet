import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import aggr
import sys
import traceback

sqrt_eps = 1e-12


class InfluencerLoss(nn.Module):
    def __init__(
        self,
        user_influencer_weight=1.0,
        influencer_influencer_weight=1.0,
        user_influencer_neg_ratio=1.0,
        user_margin=1.0,
        influencer_margin=1.0,
        device="cpu",
        scatter_loss=False,
    ):
        super(InfluencerLoss, self).__init__()
        self.user_influencer_weight = user_influencer_weight
        self.influencer_influencer_weight = influencer_influencer_weight
        self.user_influencer_neg_ratio = user_influencer_neg_ratio
        self.user_margin = user_margin
        self.influencer_margin = influencer_margin
        self.device = device
        self.scatter_loss = scatter_loss

    def forward(
        self,
        user_embed,
        influencer_embed,
        batch,
        user_influencer_edges,
        user_influencer_truth,
        influencer_influencer_edges,
        influencer_influencer_truth,
    ):
        user_influencer_loss = self.get_user_influencer_loss(
            batch,
            user_influencer_edges,
            user_influencer_truth,
            user_embed,
            influencer_embed,
        )
        influencer_influencer_loss = self.get_influencer_influencer_loss(
            influencer_influencer_edges, influencer_influencer_truth, influencer_embed
        )

        loss = (
            self.user_influencer_weight * user_influencer_loss
            + self.influencer_influencer_weight * influencer_influencer_loss
        )
        return loss

    def get_user_influencer_loss(
        self,
        batch,
        user_influencer_edges,
        user_influencer_truth,
        user_embed,
        influencer_embed,
    ):
        return get_user_influencer_loss(
            batch,
            user_influencer_edges,
            user_influencer_truth,
            user_embed,
            influencer_embed,
            self.user_influencer_weight,
            self.user_influencer_neg_ratio,
            self.margin,
            self.device,
            self.scatter_loss,
        )

    def get_influencer_influencer_loss(
        self, influencer_influencer_edges, influencer_influencer_truth, influencer_embed
    ):
        return get_influencer_influencer_loss(
            influencer_influencer_edges,
            influencer_influencer_truth,
            influencer_embed,
            self.influencer_influencer_weight,
            self.influencer_margin,
            self.device,
        )


mean_agg = aggr.MeanAggregation()


def geometric_mean(x, dim=0):
    return torch.exp(torch.mean(torch.log(x), dim=dim))


def get_user_influencer_loss(
    batch,
    user_influencer_edges,
    user_influencer_truth,
    user_embed,
    influencer_embed,
    user_influencer_weight,
    user_influencer_neg_ratio,
    user_margin,
    device,
    scatter_loss,
):
    if not scatter_loss:
        positive_loss = []
        pid, particle_length = batch.pid.unique(return_counts=True)
        # TODO: HANDLE THIS CHECK IN THE FUNCTION/CLASS WAY!
        if particle_length.max() > 30:
            print(
                "Too many particles in batch",
                particle_length[particle_length > 30],
                pid[particle_length > 30],
            )

        for pid, particle_length in torch.stack(batch.pid.unique(return_counts=True)).T:
            true_hits = torch.where(batch.pid == pid)[0]
            true_mesh = torch.meshgrid(true_hits, true_hits)

            dist_sq = torch.sum(
                (user_embed[true_mesh[0]] - influencer_embed[true_mesh[1]]) ** 2, dim=-1
            )  # SQREUCLIDEAN
            dist = torch.sqrt(dist_sq + sqrt_eps)  # EUCLIDEAN
            follower_sum = dist.mean(dim=0)  # / user_margin # EUCLIDEAN
            influencer_prod = geometric_mean(follower_sum, dim=0)

            # Check if influencer_prod is nan or inf
            try:
                assert not torch.isnan(influencer_prod) and not torch.isinf(influencer_prod)
            except AssertionError:
                print(
                    f"Influencer prod is nan or inf... \n Influencer prod: {influencer_prod} \n pid: {pid} \n particle_length: {particle_length} \n dist_sq: {dist_sq} \n dist: {dist} \n follower_sum: {follower_sum} \n user_embed: {user_embed[true_mesh[0]]} \n influencer_embed: {influencer_embed[true_mesh[1]]}"
                )
                sys.exit()

            positive_loss.append(influencer_prod)
        positive_loss = torch.stack(positive_loss).mean()

    else:
        # Sparse version
        dist_sq = torch.sum(
            (user_embed[batch.edge_index[0]] - influencer_embed[batch.edge_index[1]])
            ** 2,
            dim=-1,
        )  # SQREUCLIDEAN
        # dist = torch.sqrt(dist_sq + sqrt_eps) # EUCLIDEAN
        # follower_sum = mean_agg(dist, batch.edge_index[1], dim_size=batch.pid.shape[0], dim=0) / user_margin # EUCLIDEAN
        follower_sum = (
            mean_agg(dist_sq, batch.edge_index[1], dim_size=batch.pid.shape[0], dim=0)
            / user_margin**2
        )  # SQREUCLIDEAN
        follower_sum = torch.log(follower_sum)
        # Get an indexable PID for each node. Currently the PID of each node (batch.pid) is a unique number for each particle, but not necessarily a continuous index.
        _, inverse_indices = torch.unique(batch.pid, return_inverse=True)
        influencer_prod = mean_agg(follower_sum, inverse_indices, dim=0)
        influencer_prod = torch.exp(influencer_prod)
        positive_loss = influencer_prod.mean()

    hinge, d = get_hinge_distance(
        user_embed, influencer_embed, user_influencer_edges, user_influencer_truth
    )

    if (hinge == -1).any():
        negative_loss = (
            torch.stack(
                [user_margin - d[hinge == -1], torch.zeros_like(d[hinge == -1])], dim=1
            )
            .max(dim=1)[0]
            .pow(2)
            .mean()
        )
    else:
        negative_loss = torch.tensor(0, dtype=torch.float32, device=device)

    loss = torch.tensor(0, dtype=torch.float32, device=device)
    if not torch.isnan(negative_loss):
        loss += user_influencer_neg_ratio * negative_loss
    if not torch.isnan(positive_loss):
        loss += positive_loss
    return user_influencer_weight * loss


def get_influencer_influencer_loss(
    influencer_influencer_edges,
    influencer_influencer_truth,
    influencer_embed,
    influencer_influencer_weight,
    influencer_margin,
    device,
    loss_type="hinge",
):
    _, d = get_hinge_distance(
        influencer_embed,
        influencer_embed,
        influencer_influencer_edges,
        influencer_influencer_truth,
    )

    if loss_type == "hinge":
        loss = (
            torch.stack([influencer_margin - d, torch.zeros_like(d)], dim=1)
            .max(dim=1)[0]
            .pow(2)
            .mean()
        )
    elif loss_type == "semilinear":
        loss = (
            torch.stack([influencer_margin - d, torch.zeros_like(d)], dim=1)
            .max(dim=1)[0]
            .mean()
        )
    elif loss_type == "inverse":
        loss = (1.0 / (d + 1e-12)).mean()
    elif loss_type == "hyperbolic":
        loss_input = 1.0 / (d * (2 * influencer_margin - d) + 1e-12) - (
            1 / influencer_margin**2
        )
        loss_input[d >= influencer_margin] = 0.0
        loss = loss_input.mean()
    else:
        raise NotImplementedError

    return influencer_influencer_weight * loss


def get_hinge_distance(query, database, edges, truth, p=1):
    hinge = truth.float()
    hinge[hinge == 0] = -1
    if p == 1:
        d = torch.sqrt(
            torch.sum((query[edges[0]] - database[edges[1]]) ** 2, dim=-1) + sqrt_eps
        )  # EUCLIDEAN
    elif p == 2:
        d = torch.sum(
            (query[edges[0]] - database[edges[1]]) ** 2, dim=-1
        )  # SQR-EUCLIDEAN
    else:
        raise NotImplementedError

    return hinge, d


def influencer_loss(
    user_embed,
    influencer_embed,
    batch,
    user_influencer_edges,
    user_influencer_truth,
    influencer_influencer_edges,
    influencer_influencer_truth,
    user_influencer_weight=1.0,
    influencer_influencer_weight=1.0,
    user_influencer_neg_ratio=1.0,
    user_margin=1.0,
    influencer_margin=1.0,
    device="cpu",
    scatter_loss=False,
    loss_type="hinge",
):
    # Initialize losses to zero
    user_influencer_loss = torch.tensor(0.0, device=device)
    influencer_influencer_loss = torch.tensor(0.0, device=device)

    # Calculate each loss component if edges are not empty
    if user_influencer_edges.nelement() != 0:
        user_influencer_loss = get_user_influencer_loss(
            batch,
            user_influencer_edges,
            user_influencer_truth,
            user_embed,
            influencer_embed,
            user_influencer_weight,
            user_influencer_neg_ratio,
            user_margin,
            device,
            scatter_loss,
        )

    if influencer_influencer_edges.nelement() != 0:
        influencer_influencer_loss = get_influencer_influencer_loss(
            influencer_influencer_edges,
            influencer_influencer_truth,
            influencer_embed,
            influencer_influencer_weight,
            influencer_margin,
            device,
            loss_type=loss_type,
        )

    # Compute the total loss
    loss = (
        user_influencer_weight * user_influencer_loss
        + influencer_influencer_weight * influencer_influencer_loss
    )

    sublosses = {
        "user_influencer_loss": user_influencer_loss.item(),
        "influencer_influencer_loss": influencer_influencer_loss.item(),
    }
    return loss, sublosses
