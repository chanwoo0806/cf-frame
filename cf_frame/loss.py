import torch
import torch.nn.functional as F
from cf_frame.util import pload, pstore
from cf_frame.configurator import args


class BPR:
    def __init__(self):
        self.embed_reg = args.embed_reg
        self.type = 'pairwise'
        
    def __call__(self, model, batch_data):
        ancs, poss, negs = batch_data
        
        # compute BPR loss
        user_embeds, item_embeds = model.forward()
        anc_embeds, pos_embeds, neg_embeds = user_embeds[ancs], item_embeds[poss], item_embeds[negs]
        pos_preds = (anc_embeds * pos_embeds).sum(dim=-1)
        neg_preds = (anc_embeds * neg_embeds).sum(dim=-1)
        bpr_loss = torch.sum(F.softplus(neg_preds - pos_preds)) / len(ancs)
        
        # compute regularization loss
        anc_egos, pos_egos, neg_egos = model.user_embeds[ancs], model.item_embeds[poss], model.item_embeds[negs]
        reg_loss = (anc_egos.norm(2).pow(2) + pos_egos.norm(2).pow(2) + neg_egos.norm(2).pow(2)) / len(ancs)
        reg_loss *= self.embed_reg * 0.5
        
        loss = bpr_loss + reg_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss}
        return loss, losses
    
class DirectAU:
    def __init__(self):
        self.gamma = args.gamma
        self.type = 'pairwise' # actually, it's pointwise
        
    def _alignment(self, x, y, alpha=2):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()
    
    def _uniformity(self, x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()
    
    def __call__(self, model, batch_data):
        ancs, poss, _ = batch_data
        
        # compute alignment and uniformity loss
        user_embeds, item_embeds = model.forward()
        anc_embeds, pos_embeds = user_embeds[ancs], item_embeds[poss]
        align = self._alignment(anc_embeds, pos_embeds)
        uniform = 0.5 * (self._uniformity(anc_embeds) + self._uniformity(pos_embeds))
        
        loss = align + self.gamma * uniform
        losses = {'align': align, 'uniform': uniform}
        return loss, losses


# Infinite Layer Skip for UltraGCN
class LayerSkipLoss:
    def __init__(self):
        # self.user_num = args.user_num
        # self.item_num = args.item_num
        # self.embedding_dim = args.embed_dim
        self.w1 = args.w1
        self.w2 = args.w2
        self.w3 = args.w3
        self.w4 = args.w4

        self.negative_weight = args.negative_weight
        self.gamma = args.gamma
        self.lambda_ = args.lambda_

        constraint_mat_path = f'./dataset/{args.dataset}/constraint_mat.pkl'
        ii_constraint_mat_path = f'./dataset/{args.dataset}/ii_constraint_mat.pkl'
        ii_neighbor_mat_path = f'./dataset/{args.dataset}/ii_neighbor_mat.pkl'

        self.constraint_mat = pload(constraint_mat_path)
        self.ii_constraint_mat = pload(ii_constraint_mat_path)
        self.ii_neighbor_mat = pload(ii_neighbor_mat_path)

        self.type = 'multineg'

    def get_omegas(self, users, pos_items, neg_items):
        device = self.get_device()
        if self.w2 > 0:
            pos_weight = torch.mul(self.constraint_mat['beta_uD'][users], self.constraint_mat['beta_iD'][pos_items]).to(device)
            pos_weight = self.w1 + self.w2 * pos_weight
        else:
            pos_weight = self.w1 * torch.ones(len(pos_items)).to(device)
        
        # users = (users * self.item_num).unsqueeze(0)
        if self.w4 > 0:
            neg_weight = torch.mul(torch.repeat_interleave(self.constraint_mat['beta_uD'][users], neg_items.size(1)), self.constraint_mat['beta_iD'][neg_items.flatten()]).to(device)
            neg_weight = self.w3 + self.w4 * neg_weight
        else:
            neg_weight = self.w3 * torch.ones(neg_items.size(0) * neg_items.size(1)).to(device)


        weight = torch.cat((pos_weight, neg_weight))
        return weight
    
    def cal_loss_L(self, users, pos_items, neg_items, omega_weight):
        device = self.get_device()
        user_embeds = self.user_embeds[users]
        pos_embeds = self.item_embeds[pos_items]
        neg_embeds = self.item_embeds[neg_items]
      
        pos_scores = (user_embeds * pos_embeds).sum(dim=-1) # batch_size
        user_embeds = user_embeds.unsqueeze(1)
        neg_scores = (user_embeds * neg_embeds).sum(dim=-1) # batch_size * negative_num

        neg_labels = torch.zeros(neg_scores.size()).to(device)
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_labels, weight = omega_weight[len(pos_scores):].view(neg_scores.size()), reduction='none').mean(dim = -1)
        
        pos_labels = torch.ones(pos_scores.size()).to(device)
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels, weight = omega_weight[:len(pos_scores)], reduction='none')

        loss = pos_loss + neg_loss * self.negative_weight
      
        return loss.sum()
    
    def cal_loss_I(self, users, pos_items):
        device = self.get_device()
        neighbor_embeds = self.item_embeds[self.ii_neighbor_mat[pos_items].to(device)]    # len(pos_items) * num_neighbors * dim
        sim_scores = self.ii_constraint_mat[pos_items].to(device)     # len(pos_items) * num_neighbors
        user_embeds = self.user_embeds[users].unsqueeze(1)
        
        loss = -sim_scores * (user_embeds * neighbor_embeds).sum(dim=-1).sigmoid().log()
      
        # loss = loss.sum(-1)
        return loss.sum()
    
    def norm_loss(self, model):
        loss = 0.0
        for parameter in model.parameters():
            loss += torch.sum(parameter ** 2)
        return loss / 2
    
    def get_device(self):
        # return self.user_embeds.weight.device
        return args.device
    
    # def forward(self, users, pos_items, neg_items):
    #     omega_weight = self.get_omegas(users, pos_items, neg_items)
        
    #     loss = self.cal_loss_L(users, pos_items, neg_items, omega_weight)
    #     loss += self.gamma * self.norm_loss()
    #     loss += self.lambda_ * self.cal_loss_I(users, pos_items)
    #     return loss

    # def test_foward(self, users):
    #     items = torch.arange(self.item_num).to(users.device)
    #     user_embeds = self.user_embeds(users)
    #     item_embeds = self.item_embeds(items)
         
    #     return user_embeds.mm(item_embeds.t())
    
    def __call__(self, model, batch_data):
        users, pos_items, neg_items = batch_data
        self.user_embeds, self.item_embeds = model.forward()
        omega_weight = self.get_omegas(users, pos_items, neg_items)

        loss = 0
        loss_L = self.cal_loss_L(users, pos_items, neg_items, omega_weight)
        loss_norm = self.norm_loss(model)
        loss_I = self.cal_loss_I(users, pos_items)

        loss = loss_L + self.gamma * loss_norm + self.lambda_ * loss_I
        losses = {
            'loss': loss, 'loss_I': loss_I, 'loss_L': loss_L, 'loss_norm': loss_norm
        }
        return loss, losses