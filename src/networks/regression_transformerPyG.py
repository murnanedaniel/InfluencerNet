# 3rd party imports
import torch
from torch import nn
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import aggr
from torch_geometric.nn import TransformerConv

# Local imports
from .utils import make_mlp


class RegressionTransformerPyG(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        hparams["regression_dim"] = len(hparams["regression_targets"])

        self.input_network = make_mlp(
            hparams["spatial_channels"],
            [hparams["emb_hidden"]] * hparams["nb_layer"],
            hidden_activation=hparams["activation"],
            output_activation=hparams["activation"],
            layer_norm=True,
        )

        hparams["nb_transformer_layers"] = (
            hparams["nb_layer"]
            if "nb_transformer_layers" not in hparams
            else hparams["nb_transformer_layers"]
        )

        self.transformer_layers = nn.ModuleList(
            [
                TransformerConv(
                    hparams["emb_hidden"]
                    if i == 0
                    else hparams["num_heads"] * hparams["emb_hidden"],
                    hparams["emb_hidden"],
                    heads=hparams["num_heads"],
                    dropout=0.0,
                )
                for i in range(hparams["nb_transformer_layers"])
            ]
        )

        aggrs = ["sum", "mean", "min", "max", "std"]

        self.regression_network = make_mlp(
            hparams["num_heads"] * hparams["emb_hidden"] * len(aggrs),
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["regression_dim"]],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=True,
        )

        self.aggr = aggr.MultiAggregation(
            aggrs=aggrs,
        )

    def forward(self, x, batch=None, edge_index=None):
        x = self.input_network(x)

        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, edge_index)

        track_level_x = self.aggr(x, batch)

        regression_out = self.regression_network(track_level_x)

        return regression_out
