# 3rd party imports
import torch
from torch import nn
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.pool import global_mean_pool

# Local imports
from .utils import make_mlp


class RegressionTransformer(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """

        hparams["regression_dim"] = len(hparams["regression_targets"])

        self.input_network = make_mlp(
            hparams["spatial_channels"],
            [hparams["emb_hidden"]] * hparams["nb_layer"],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=True,
        )

        hparams["nb_transformer_layers"] = (
            hparams["nb_layer"]
            if "nb_transformer_layers" not in hparams
            else hparams["nb_transformer_layers"]
        )

        transformer_activation = "relu" if hparams["activation"] == "ReLU" else "gelu"
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hparams["emb_hidden"],
            nhead=hparams["num_heads"],
            dim_feedforward=hparams["emb_hidden"],
            dropout=0.0,
            activation=transformer_activation,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer, num_layers=hparams["nb_transformer_layers"]
        )

        self.regression_network = make_mlp(
            hparams["emb_hidden"],
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["regression_dim"]],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=True,
        )

    def forward(self, x, batch=None, **kwargs):
        x = self.input_network(x)
        if batch is not None:
            x, mask = to_dense_batch(x, batch)
            x = self.transformer_encoder(x, src_key_padding_mask=(~mask))
            x = x[mask]
        else:
            x = self.transformer_encoder(x.unsqueeze(0)).squeeze(0)

        track_level_x = global_mean_pool(x, batch)

        regression_out = self.regression_network(track_level_x)

        return regression_out
