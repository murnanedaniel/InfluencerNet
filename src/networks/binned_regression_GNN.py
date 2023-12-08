# 3rd party imports
import torch
from torch import nn
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import aggr
from torch_geometric.nn import TransformerConv

# Local imports
from .utils import make_mlp

class BinnedRegressionInteractionGNN(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        hparams["regression_dim"] = len(hparams["regression_targets"])

        # Setup input network
        self.node_encoder = make_mlp(
            hparams["spatial_channels"],
            [hparams["emb_hidden"]] * hparams["nb_layer"],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=True,
        )

        # The edge network computes new edge features from connected nodes
        self.edge_encoder = make_mlp(
            2 * hparams["emb_hidden"],
            [hparams["emb_hidden"]] * hparams["nb_layer"],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=True,
        )

        # The node network computes new node features
        self.node_networks = nn.ModuleList(
            [
                make_mlp(
                    2 * hparams["emb_hidden"],
                    [hparams["emb_hidden"]] * hparams["nb_layer"],
                    hidden_activation=hparams["activation"],
                    output_activation=None,
                    layer_norm=True,
                )
                for _ in range(hparams["nb_transformer_layers"])
            ]
        )

        # The edge network computes new edge features
        self.edge_networks = nn.ModuleList(
            [
                make_mlp(
                    3 * hparams["emb_hidden"],
                    [hparams["emb_hidden"]] * hparams["nb_layer"],
                    hidden_activation=hparams["activation"],
                    output_activation=None,
                    layer_norm=True,
                )
                for _ in range(hparams["nb_transformer_layers"])
            ]
        )

        aggrs = ["sum", "mean"]

        self.regression_network = make_mlp(
            hparams["emb_hidden"] * hparams["nb_transformer_layers"] * len(aggrs),
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["num_bins"]],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=True,
        )

        # Instead of torch scatter sum, we use the torch aggr class
        self.node_aggr = aggr.MultiAggregation(
            aggrs = ["sum"],
        )

        self.track_aggr = aggr.MultiAggregation(
            aggrs = aggrs,
        )

    def forward(self, x, batch=None, edge_index=None):
        x = self.node_encoder(x)

        start, end = edge_index
        e = self.edge_encoder(torch.cat([x[start], x[end]], dim=1))

        # Initialize a list to store node outputs
        node_outputs = []

        # Loop over iterations of edge and node networks
        for node_network, edge_network in zip(self.node_networks, self.edge_networks):
            edge_messages = self.node_aggr(e, end, dim=0, dim_size=x.shape[0])

            node_inputs = torch.cat([x, edge_messages], dim=-1)
            x = node_network(node_inputs)

            # Store the node output
            node_outputs.append(x)

            edge_inputs = torch.cat([x[start], x[end], e], dim=-1)
            e = edge_network(edge_inputs)

        # Concatenate all node outputs
        x = torch.cat(node_outputs, dim=-1)

        track_level_x = self.track_aggr(x, batch)

        return self.regression_network(track_level_x)