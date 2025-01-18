import torch
from torch import nn
from cf_frame.model import BaseModel
from cf_frame.configurator import args
from cf_frame.util import scipy_coo_to_torch_sparse


class NeuMF(BaseModel):
    def __init__(self, data_handler):
        super().__init__(data_handler)

        # Arguments
        self.embed_dim = args.embed_dim
        self.layer_num = args.layer_num
        self.dropout = args.dropout

        # Learnable Weights
        self.embed_user_gmf = nn.Embedding(self.user_num, self.embed_dim)
        self.embed_item_gmf = nn.Embedding(self.item_num, self.embed_dim)
        self.embed_user_mlp = nn.Embedding(self.user_num, self.embed_dim * (2 ** (self.layer_num - 1)))
        self.embed_item_mlp = nn.Embedding(self.item_num, self.embed_dim * (2 ** (self.layer_num - 1)))

        # mlp Module
        mlp_modules = []
        for i in range(self.layer_num):
            input_size = self.embed_dim * (2 ** (self.layer_num - i))
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, input_size // 2))
            mlp_modules.append(nn.ReLU())
        self.mlp_layers = nn.Sequential(*mlp_modules)

        # Prediction FC-layer
        self.predict_layer = nn.Linear(self.embed_dim * 2, 1)
        self.__init_weight()
    
    def __init_weight(self):
        nn.init.normal_(self.embed_user_gmf.weight, std=0.01)
        nn.init.normal_(self.embed_user_mlp.weight, std=0.01)
        nn.init.normal_(self.embed_item_gmf.weight, std=0.01)
        nn.init.normal_(self.embed_item_mlp.weight, std=0.01)

        for m in self.mlp_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, user, item):
        embed_user_gmf = self.embed_user_gmf(user)
        embed_item_gmf = self.embed_item_gmf(item)
        output_gmf = embed_user_gmf * embed_item_gmf

        embed_user_mlp = self.embed_user_mlp(user)
        embed_item_mlp = self.embed_item_mlp(item)
        interaction = torch.cat((embed_user_mlp, embed_item_mlp), -1)
        output_mlp = self.mlp_layers(interaction)

        concat = torch.cat((output_gmf, output_mlp), -1)
        prediction = self.predict_layer(concat)
        return prediction.view(-1)
    
    def full_predict(self, batch_data):
        pck_users, train_mask = batch_data

        # Get the number of users in the batch and the total number of items
        B = pck_users.size(0)  # Batch size (number of users in the batch)
        N = self.embed_item_gmf.num_embeddings  # Total number of items

        # Expand user embeddings to match the total number of items
        users = pck_users.unsqueeze(1).expand(-1, N).reshape(-1)  # Shape: (B * N)
        items = torch.arange(N, device=pck_users.device).repeat(B)  # Shape: (B * N)

        # Compute predictions for all user-item pairs in the batch
        embed_user_gmf = self.embed_user_gmf(users)  # Shape: (B * N, embed_dim)
        embed_item_gmf = self.embed_item_gmf(items)  # Shape: (B * N, embed_dim)
        output_gmf = embed_user_gmf * embed_item_gmf

        embed_user_mlp = self.embed_user_mlp(users)  # Shape: (B * N, mlp_input_dim)
        embed_item_mlp = self.embed_item_mlp(items)  # Shape: (B * N, mlp_input_dim)
        interaction = torch.cat((embed_user_mlp, embed_item_mlp), -1)  # Shape: (B * N, mlp_input_dim * 2)
        output_mlp = self.mlp_layers(interaction)  # Shape: (B * N, embed_dim)

        concat = torch.cat((output_gmf, output_mlp), -1)  # Shape: (B * N, embed_dim * 2)
        predictions = self.predict_layer(concat).view(B, N)  # Shape: (B, N)

        # Mask the predictions using train_mask
        return self._mask_predict(predictions, train_mask)