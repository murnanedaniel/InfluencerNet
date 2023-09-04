# 3rd party imports
from ..influencer_base import InfluencerBase
import torch.nn.functional as F
from torch import nn
import torch
import copy
from torch_geometric.utils import to_dense_batch

# Local imports
from ..utils import make_mlp

class InfluencerDummy(InfluencerBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """

        self.setup(stage="fit")

        self.valset = self.trainset

        # Get the 2D positions of the nodes from the trainset and make them the trainable parameters of the model
        self.users = nn.Parameter(self.trainset[0].x)
        self.influencers = nn.Parameter(self.trainset[0].x)

        self.save_hyperparameters()

    def forward(self, x, batch=None):
        
        return self.users, self.influencers

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
        user_embed, influencer_embed = self(input_data, batch.batch)

        # Get the training edges for each loss function
        user_user_edges, user_user_truth = self.get_training_edges(batch, user_embed, user_embed, hnm=True, rp=True, tp=True, batch_index = batch.batch)
        user_influencer_edges, user_influencer_truth = self.get_training_edges(batch, user_embed, influencer_embed, hnm=True, tp=True, rp=True, batch_index = batch.batch)
        influencer_influencer_edges, influencer_influencer_truth = self.get_training_edges(batch, influencer_embed, influencer_embed, hnm=True, rp=True, radius=self.hparams["influencer_margin"], batch_index = batch.batch)

        # Calculate each loss function
        user_user_loss = self.get_user_user_loss(user_user_edges, user_user_truth, user_embed)
        user_influencer_loss = self.get_user_influencer_loss(user_influencer_edges, user_influencer_truth, user_embed, influencer_embed, batch)
        influencer_influencer_loss = self.get_influencer_influencer_loss(influencer_influencer_edges, influencer_influencer_truth, influencer_embed)

        loss = user_user_loss + user_influencer_loss + influencer_influencer_loss

        self.log_dict({"train_loss": loss, "train_user_user_loss": user_user_loss, "train_user_influencer_loss": user_influencer_loss, "train_influencer_influencer_loss": influencer_influencer_loss})
        
        # print(f"User user loss: {user_user_loss} \n User influencer loss: {user_influencer_loss} \n Influencer influencer loss: {influencer_influencer_loss} \n Total loss: {loss}")
        # print(f"User user edges: {user_user_edges} \n User influencer edges: {user_influencer_edges} \n Influencer influencer edges: {influencer_influencer_edges}")
        # print(f"User user truth: {user_user_truth} \n User influencer truth: {user_influencer_truth} \n Influencer influencer truth: {influencer_influencer_truth}")

        if torch.isnan(loss):
            print("Loss is nan")
            sys.exit()

        return loss