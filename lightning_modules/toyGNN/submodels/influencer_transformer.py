# 3rd party imports
from ..influencer_base import InfluencerBase
import torch.nn.functional as F
from torch import nn
import torch
import copy
from torch_geometric.utils import to_dense_batch

# Local imports
from ..utils import make_mlp

class InfluencerTransformer(InfluencerBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """

        self.input_network = make_mlp(
            hparams["spatial_channels"],
            [hparams["emb_hidden"]],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=True,
        )

        transformer_activation = "relu" if hparams["activation"] == "ReLU" else "gelu"
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=hparams["emb_hidden"], nhead=hparams["num_heads"], dim_feedforward=hparams["emb_hidden"], dropout=0.0, activation=transformer_activation, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=hparams["nb_layer"])

        self.user_network = make_mlp(
            hparams["emb_hidden"],
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=True,
        )

        self.influencer_network = make_mlp(
            hparams["emb_hidden"],
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=True,
        )

        # Add a trainable parameter that represents the radius of the hypersphere
        self.radius = torch.nn.Parameter(torch.tensor(1.0))

        self.save_hyperparameters()

    def forward(self, x, batch=None):
        
        x = self.input_network(x)
        if batch is not None:
            x, mask = to_dense_batch(x, batch)
            x = self.transformer_encoder(x, src_key_padding_mask=(~mask))
            x = x[mask]        
        else:
            x = self.transformer_encoder(x.unsqueeze(0)).squeeze(0)

        user_out = self.user_network(x)
        influencer_out = self.influencer_network(x)

        if "norm" in self.hparams["regime"]:
            user_out = F.normalize(user_out)
            influencer_out = F.normalize(influencer_out)

            if "radius" in self.hparams["regime"]:
                user_out = self.radius * user_out
                influencer_out = self.radius * influencer_out

        return user_out, influencer_out