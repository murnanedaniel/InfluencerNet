# 3rd party imports
from ..influencer_base import InfluencerBase
import torch.nn.functional as F
import torch
import copy

# Local imports
from ..utils import make_mlp

class InfluencerEmbedding(InfluencerBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """
        self.user_network = make_mlp(
            hparams["spatial_channels"],
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=True,
        )

        # Make influencer network, a deep copy of the user network
        # self.influencer_network = copy.deepcopy(self.user_network)        

        self.influencer_network = make_mlp(
            hparams["spatial_channels"],
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=True,
        )

        # Add a trainable parameter that represents the radius of the hypersphere
        self.radius = torch.nn.Parameter(torch.tensor(1.0))

        self.save_hyperparameters()

    def forward(self, x):

        user_out = self.user_network(x)
        influencer_out = self.influencer_network(x)

        # return F.normalize(user_out) if "norm" in self.hparams["regime"] else user_out, F.normalize(influencer_out) if "norm" in self.hparams["regime"] else influencer_out
        if "norm" in self.hparams["regime"]:
            user_out = self.radius * F.normalize(user_out)
            influencer_out = self.radius * F.normalize(influencer_out)
        
        return user_out, influencer_out