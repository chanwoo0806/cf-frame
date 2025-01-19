import torch
from torch import nn
from torch.nn import functional as F
from cf_frame.model import BaseModel
from cf_frame.configurator import args


class MultiVAE(BaseModel):
    def __init__(self, data_handler):
        super().__init__(data_handler)

        # Hyperparams
        self.layers = args.mlp_hidden_size
        self.lat_dim = args.latent_dimension
        self.drop_out = args.dropout_prob

        # User-item Interaction
        self.R = data_handler.get_inter()  # Equal to rating_matrix in RecBole implementation.

        # Encoder, Decoder of VAE
        self.encode_layer_dims = [self.item_num] + self.layers + [self.lat_dim]
        self.decode_layer_dims = [int(self.lat_dim / 2)] + self.encode_layer_dims[::-1][1:]
        self.encoder = self.mlp_layers(self.encode_layer_dims)
        self.decoder = self.mlp_layers(self.decode_layer_dims)

    def mlp_layers(self, layer_dims):
        mlp_modules = []
        for i, (d_in, d_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            mlp_modules.append(nn.Linear(d_in, d_out))
            if i != len(layer_dims[:-1]) - 1:
                mlp_modules.append(nn.Tanh())
        return nn.Sequential(*mlp_modules)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.zeros_like(std).normal_(mean=0, std=0.01)
            return mu + epsilon * std
        else:
            return mu
        
    def forward(self, user):
        rating_matrix = self.R[user]
        h = F.normalize(rating_matrix)
        h = F.dropout(h, self.drop_out, training=self.training)
        h = self.encoder(h)
        mu = h[:, : int(self.lat_dim / 2)]
        logvar = h[:, int(self.lat_dim / 2) :]
        z = self.reparameterize(mu, logvar)
        z = self.decoder(z)
        return z, mu, logvar
    
    def full_predict(self, batch_data):
        pck_users, train_mask = batch_data
        scores, _, _ = self.forward(pck_users)
        return self._mask_predict(scores, train_mask)
    