import torch
from torch_geometric.nn import aggr
from torch_geometric.data import Batch

EPSILON = 1e-12
DEFAULT_DIMENSION = 0
mean_agg = aggr.MeanAggregation()

def calculate_sparse_positive_loss(batch: Batch, follower_embed: torch.Tensor, influencer_embed: torch.Tensor, follower_margin: float) -> torch.Tensor:
    """
    Calculate the sparse positive loss.

    Parameters:
    batch (Batch): The batch of data.
    follower_embed (torch.Tensor): The follower embeddings.
    influencer_embed (torch.Tensor): The influencer embeddings.
    follower_margin (float): The follower margin.

    Returns:
    torch.Tensor: The calculated sparse positive loss.
    """
    dist_sq = torch.sum((follower_embed[batch.edge_index[0]] - influencer_embed[batch.edge_index[1]]) ** 2, dim=-1)
    follower_sum = mean_agg(dist_sq, batch.edge_index[1], dim_size=batch.pid.shape[0], dim=0) / follower_margin**2
    follower_sum = torch.log(follower_sum)
    _, inverse_indices = torch.unique(batch.pid, return_inverse=True)
    influencer_prod = mean_agg(follower_sum, inverse_indices, dim=0)
    return torch.exp(influencer_prod).mean()

def get_follower_influencer_loss(
    batch: Batch,
    follower_influencer_edges: torch.Tensor,
    follower_influencer_truth: torch.Tensor,
    follower_embed: torch.Tensor,
    influencer_embed: torch.Tensor,
    follower_influencer_weight: float,
    follower_influencer_neg_ratio: float,
    follower_margin: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Calculate the follower influencer loss.

    Parameters:
    batch (Batch): The batch of data.
    follower_influencer_edges (torch.Tensor): The follower influencer edges.
    follower_influencer_truth (torch.Tensor): The follower influencer truth values.
    follower_embed (torch.Tensor): The follower embeddings.
    influencer_embed (torch.Tensor): The influencer embeddings.
    follower_influencer_weight (float): The follower influencer weight.
    follower_influencer_neg_ratio (float): The follower influencer negative ratio.
    follower_margin (float): The follower margin.
    device (torch.device): The device to use for computations.

    Returns:
    torch.Tensor: The calculated follower influencer loss.
    """
    positive_loss = calculate_sparse_positive_loss(batch, follower_embed, influencer_embed, follower_margin)
    
    hinge, d = get_hinge_distance(
        follower_embed, influencer_embed, follower_influencer_edges, follower_influencer_truth
    )

    if (hinge == -1).any():
        negative_loss = (
            torch.stack(
                [follower_margin - d[hinge == -1], torch.zeros_like(d[hinge == -1])], dim=1
            )
            .max(dim=1)[0]
            .pow(2)
            .mean()
        )
    else:
        negative_loss = torch.tensor(0, dtype=torch.float32, device=device)

    loss = torch.tensor(0, dtype=torch.float32, device=device)
    if not torch.isnan(negative_loss):
        loss += follower_influencer_neg_ratio * negative_loss
    if not torch.isnan(positive_loss):
        loss += positive_loss
    return follower_influencer_weight * loss

def get_influencer_influencer_loss(
    influencer_influencer_edges: torch.Tensor,
    influencer_influencer_truth: torch.Tensor,
    influencer_embed: torch.Tensor,
    influencer_influencer_weight: float,
    influencer_margin: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Calculate the influencer influencer loss.

    Parameters:
    influencer_influencer_edges (torch.Tensor): The influencer influencer edges.
    influencer_influencer_truth (torch.Tensor): The influencer influencer truth values.
    influencer_embed (torch.Tensor): The influencer embeddings.
    influencer_influencer_weight (float): The influencer influencer weight.
    influencer_margin (float): The influencer margin.
    device (torch.device): The device to use for computations.

    Returns:
    torch.Tensor: The calculated influencer influencer loss.
    """
    _, d = get_hinge_distance(
        influencer_embed,
        influencer_embed,
        influencer_influencer_edges,
        influencer_influencer_truth,
    )

    loss = (
            torch.stack([influencer_margin - d, torch.zeros_like(d)], dim=1)
            .max(dim=1)[0].mean()
        )

    return influencer_influencer_weight * loss

def get_hinge_distance(query: torch.Tensor, database: torch.Tensor, edges: torch.Tensor, truth: torch.Tensor, p: int = 1) -> torch.Tensor:
    """
    Calculate the hinge distance.

    Parameters:
    query (torch.Tensor): The query tensor.
    database (torch.Tensor): The database tensor.
    edges (torch.Tensor): The edges tensor.
    truth (torch.Tensor): The truth tensor.
    p (int): The power for distance calculation. Default is 1.

    Returns:
    torch.Tensor: The calculated hinge distance.
    """
    hinge = truth.float()
    hinge[hinge == 0] = -1
    if p == 1:
        d = torch.sqrt(
            torch.sum((query[edges[0]] - database[edges[1]]) ** 2, dim=-1) + EPSILON
        )  # EUCLIDEAN
    elif p == 2:
        d = torch.sum(
            (query[edges[0]] - database[edges[1]]) ** 2, dim=-1
        )  # SQR-EUCLIDEAN
    else:
        raise ValueError("p should be either 1 or 2")

    return hinge, d

def influencer_loss(
    follower_embed: torch.Tensor,
    influencer_embed: torch.Tensor,
    batch: Batch,
    follower_influencer_edges: torch.Tensor,
    follower_influencer_truth: torch.Tensor,
    influencer_influencer_edges: torch.Tensor,
    influencer_influencer_truth: torch.Tensor,
    follower_influencer_weight: float = 1.0,
    influencer_influencer_weight: float = 1.0,
    follower_influencer_neg_ratio: float = 1.0,
    follower_margin: float = 1.0,
    influencer_margin: float = 1.0,
) -> torch.Tensor:
    """
    Calculate the influencer loss.

    Parameters:
    follower_embed (torch.Tensor): The follower embeddings.
    influencer_embed (torch.Tensor): The influencer embeddings.
    batch (Batch): The batch of data.
    follower_influencer_edges (torch.Tensor): The follower influencer edges.
    follower_influencer_truth (torch.Tensor): The follower influencer truth values.
    influencer_influencer_edges (torch.Tensor): The influencer influencer edges.
    influencer_influencer_truth (torch.Tensor): The influencer influencer truth values.
    follower_influencer_weight (float): The follower influencer weight. Default is 1.0.
    influencer_influencer_weight (float): The influencer influencer weight. Default is 1.0.
    follower_influencer_neg_ratio (float): The follower influencer negative ratio. Default is 1.0.
    follower_margin (float): The follower margin. Default is 1.0.
    influencer_margin (float): The influencer margin. Default is 1.0.

    Returns:
    torch.Tensor: The calculated influencer loss.
    """
    device = follower_embed.device

    # Initialize losses to zero
    follower_influencer_loss = torch.tensor(0.0, device=device)
    influencer_influencer_loss = torch.tensor(0.0, device=device)

    # Calculate each loss component if edges are not empty
    if follower_influencer_edges.nelement() != 0:
        follower_influencer_loss = get_follower_influencer_loss(
            batch,
            follower_influencer_edges,
            follower_influencer_truth,
            follower_embed,
            influencer_embed,
            follower_influencer_weight,
            follower_influencer_neg_ratio,
            follower_margin,
            device,
        )

    if influencer_influencer_edges.nelement() != 0:
        influencer_influencer_loss = get_influencer_influencer_loss(
            influencer_influencer_edges,
            influencer_influencer_truth,
            influencer_embed,
            influencer_influencer_weight,
            influencer_margin,
            device,
        )

    # Compute the total loss
    loss = (
        follower_influencer_weight * follower_influencer_loss
        + influencer_influencer_weight * influencer_influencer_loss
    )

    sublosses = {
        "follower_influencer_loss": follower_influencer_loss.item(),
        "influencer_influencer_loss": influencer_influencer_loss.item(),
    }
    return loss, sublosses