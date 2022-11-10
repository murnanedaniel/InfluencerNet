# 3rd party imports
from ..embedding_base import EmbeddingBase
import torch.nn.functional as F

# Local imports
from ..utils import make_mlp

class VanillaEmbedding(EmbeddingBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """
        self.network = make_mlp(
            hparams["spatial_channels"],
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=True,
        )

        self.save_hyperparameters()

    def forward(self, x):

        x_out = self.network(x)

        return F.normalize(x_out) if "norm" in self.hparams["regime"] else x_out
