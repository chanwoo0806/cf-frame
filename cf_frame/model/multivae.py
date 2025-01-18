import torch
from torch import nn
from torch.nn import functional as F
from cf_frame.model import BaseModel
from cf_frame.configurator import args


class MultiVAE(BaseModel):
    """
    Multi-VAE: Variational Autoencoder with Multinomial Likelihood
    Adapted for our framework.
    """
    def __init__(self, data_handler):
        super().__init__(data_handler)
        self.p_dims = args.p_dims
        self.q_dims = args.q_dims if args.q_dims else self.p_dims[::-1]

        # Last layer of q-network outputs mean and logvar
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers = nn.ModuleList([
            nn.Linear(d_in, d_out) for d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])
        ])
        self.p_layers = nn.ModuleList([
            nn.Linear(d_in, d_out) for d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])
        ])

        self.drop = nn.Dropout(args.dropout)
        self._init_weights()

    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def encode(self, input):
        h = F.normalize(input)
        h = self.drop(h)

        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = torch.tanh(h)
            else:
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = torch.tanh(h)
        return h

    def _init_weights(self):
        for layer in self.q_layers + self.p_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.normal_(layer.bias, 0.0, 0.001)
