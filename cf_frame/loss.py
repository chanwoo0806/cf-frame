import torch
from torch import nn
import torch.nn.functional as F
from cf_frame.configurator import args


# Graph Signal Processing
class NonParam:
    def __init__(self):
        self.type = 'nonparam'


# MultiVAE Loss
class VAELoss:
    def __init__(self):
        self.type = 'vae'
        self.update = 0
        self.anneal_cap = args.anneal_cap
        self.total_anneal_steps = args.total_anneal_steps

    def __call__(self, model, batch_data):
        user = batch_data

        self.update += 1
        if self.total_anneal_steps > 0:
            anneal = min(self.anneal_cap, 1.0 * self.update / self.total_anneal_steps)
        else:
            anneal = self.anneal_cap

        z, mu, logvar = model.forward(user)

        kl_loss = (
            -0.5
            * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            * anneal
        )

        rating_matrix = model.R[user]
        ce_loss = -(F.log_softmax(z, 1) * rating_matrix).sum(1).mean()

        loss = ce_loss + kl_loss
        losses = {'loss': loss, 'CrossEntropy': ce_loss, 'KL-divergence': kl_loss}
        return loss, losses


# Binary Cross Entropy
class BCE:
    def __init__(self):
        self.reg_weight = args.reg_weight
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.type = 'bce'

    def __call__(self, model, batch_data):
        users, items, labels = batch_data
        preds = model.forward(users, items)
        loss = self.loss_fn(preds, labels.to(dtype=torch.float32))
        losses = {'loss': loss}
        return loss, losses


# AFDGCF (SIGIR 24)
class AFDLoss:
    def __init__(self):
        self.type = 'pairwise'
        self.reg_weight = args.reg_weight  # L2 Regularization.
        self.alpha = args.alpha  # AFD Regularization.
        self.device = args.device

    def calculate_correlation(self, emb):
        coeff = emb.T.corrcoef()
        nan_count = coeff.isnan().sum()
        if nan_count > 0:
            coeff[coeff.isnan()] = 0
        return coeff.triu(diagonal=1).norm()  

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
        reg_loss *= self.reg_weight * 0.5
        
        # compute AFD loss
        cor_loss_u = torch.zeros((1,)).to(self.device)
        cor_loss_i = torch.zeros((1,)).to(self.device)

        user_layer_correlations = []
        item_layer_correlations = []

        for i in range(1, model.layer_num+1):
            emb_u, emb_i = torch.split(model.embeds_list[i], [model.user_num, model.item_num])
            user_layer_correlations.append(self.calculate_correlation(emb_u))
            item_layer_correlations.append(self.calculate_correlation(emb_i))
        
        user_layer_correlations_coef = \
            (1 / torch.tensor(user_layer_correlations)) / torch.sum(1 / torch.tensor(user_layer_correlations))
        item_layer_correlations_coef = \
            (1 / torch.tensor(item_layer_correlations)) / torch.sum(1 / torch.tensor(item_layer_correlations))

        for i in range(1, model.layer_num+1):
            cor_loss_u += user_layer_correlations_coef[i - 1] * user_layer_correlations[i - 1]
            cor_loss_i += item_layer_correlations_coef[i - 1] * item_layer_correlations[i - 1]

        afd_loss = self.alpha * (cor_loss_u + cor_loss_i)

        loss = bpr_loss + reg_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'afd_loss': afd_loss}
        return loss, losses


# Bayesian Personalized Ranking (UAI 08)
class BPR:
    def __init__(self):
        self.reg_weight = args.reg_weight
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
        reg_loss *= self.reg_weight * 0.5
        
        loss = bpr_loss + reg_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss}
        return loss, losses
    

# LightGCL (ICLR 23)
class GCLLoss:
    def __init__(self):
        self.type = 'pairwise'
        self.temp = args.temp
        self.embed_reg = args.embed_reg
        self.cl_weight = args.cl_weight

    def __call__(self, model, batch_data):
        self.is_training = True

        user_embeds, item_embeds = model.forward()
        ancs, poss, negs = batch_data

        # compute BPR loss
        anc_embeds, pos_embeds, neg_embeds = user_embeds[ancs], item_embeds[poss], item_embeds[negs]
        pos_preds = (anc_embeds * pos_embeds).sum(dim=-1)
        neg_preds = (anc_embeds * neg_embeds).sum(dim=-1)
        bpr_loss = torch.sum(F.softplus(neg_preds - pos_preds)) / len(ancs)
        
        # compute regularization loss
        anc_egos, pos_egos, neg_egos = model.user_embeds[ancs], model.item_embeds[poss], model.item_embeds[negs]
        reg_loss = (anc_egos.norm(2).pow(2) + pos_egos.norm(2).pow(2) + neg_egos.norm(2).pow(2)) / len(ancs)
        reg_loss *= self.embed_reg * 0.5

        G_u_norm = model.G_u
        E_u_norm = model.E_u
        G_i_norm = model.G_i
        E_i_norm = model.E_i
        neg_score = torch.log(torch.exp(G_u_norm[ancs] @ E_u_norm.T / self.temp).sum(1) + 1e-8).mean()
        neg_score += torch.log(torch.exp(G_i_norm[poss] @ E_i_norm.T / self.temp).sum(1) + 1e-8).mean()
        pos_score = (torch.clamp((G_u_norm[ancs] * E_u_norm[ancs]).sum(1) / self.temp, -5.0, 5.0)).mean() + \
                    (torch.clamp((G_i_norm[poss] * E_i_norm[poss]).sum(1) / self.temp, -5.0, 5.0)).mean()
        cl_loss = -pos_score + neg_score
        cl_loss = self.cl_weight * cl_loss

        loss = bpr_loss + cl_loss + reg_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'cl_loss': cl_loss}
        return loss, losses


# DirectAU (KDD 22)
class DirectAU:
    def __init__(self):
        self.gamma = args.uniform
        self.type = 'pointwise'
        
    def _alignment(self, x, y, alpha=2):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()
    
    def _uniformity(self, x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()
    
    def __call__(self, model, batch_data):
        ancs, poss = batch_data
        
        # compute alignment and uniformity loss
        user_embeds, item_embeds = model.forward()
        anc_embeds, pos_embeds = user_embeds[ancs], item_embeds[poss]
        align = self._alignment(anc_embeds, pos_embeds)
        uniform = 0.5 * (self._uniformity(anc_embeds) + self._uniformity(pos_embeds))
        
        loss = align + self.gamma * uniform
        losses = {'align': align, 'uniform': uniform}
        return loss, losses


# UltraGCN (CIKM 21)
class LayerSkipLoss:
    def __init__(self):
        self.type = 'multineg'

        self.w1 = args.w1
        self.w2 = args.w2
        self.w3 = args.w3
        self.w4 = args.w4

        self.negative_weight = args.negative_weight
        self.gamma = args.gamma
        self.lambda_ = args.lambda_

    def get_omegas(self, users, pos_items, neg_items):
        if self.w2 > 0:
            pos_weight = torch.mul(self.constraint_mat['beta_uD'][users], self.constraint_mat['beta_iD'][pos_items]).to(args.device)
            pos_weight = self.w1 + self.w2 * pos_weight
        else:
            pos_weight = self.w1 * torch.ones(len(pos_items)).to(args.device)
        
        if self.w4 > 0:
            neg_weight = torch.mul(torch.repeat_interleave(self.constraint_mat['beta_uD'][users], neg_items.size(1)), self.constraint_mat['beta_iD'][neg_items.flatten()]).to(args.device)
            neg_weight = self.w3 + self.w4 * neg_weight
        else:
            neg_weight = self.w3 * torch.ones(neg_items.size(0) * neg_items.size(1)).to(args.device)
        weight = torch.cat((pos_weight, neg_weight))
        return weight
    
    def cal_loss_L(self, users, pos_items, neg_items, omega_weight):
        user_embeds = self.user_embeds[users]
        pos_embeds = self.item_embeds[pos_items]
        neg_embeds = self.item_embeds[neg_items]
      
        pos_scores = (user_embeds * pos_embeds).sum(dim=-1) # batch_size
        user_embeds = user_embeds.unsqueeze(1)
        neg_scores = (user_embeds * neg_embeds).sum(dim=-1) # batch_size * negative_num

        neg_labels = torch.zeros(neg_scores.size()).to(args.device)
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_labels, weight = omega_weight[len(pos_scores):].view(neg_scores.size()), reduction='none').mean(dim = -1)
        
        pos_labels = torch.ones(pos_scores.size()).to(args.device)
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels, weight = omega_weight[:len(pos_scores)], reduction='none')

        loss = pos_loss + neg_loss * self.negative_weight
        return loss.sum()
    
    def cal_loss_I(self, users, pos_items):
        neighbor_embeds = self.item_embeds[self.ii_neighbor_mat[pos_items]]    # len(pos_items) * num_neighbors * dim
        sim_scores = self.ii_constraint_mat[pos_items].to(args.device)     # len(pos_items) * num_neighbors
        user_embeds = self.user_embeds[users].unsqueeze(1)
        loss = -sim_scores * (user_embeds * neighbor_embeds).sum(dim=-1).sigmoid().log()
        return loss.sum()
    
    def norm_loss(self, model):
        loss = 0.0
        for parameter in model.parameters():
            loss += torch.sum(parameter ** 2)
        return loss / 2
    
    def __call__(self, model, batch_data):
        users, pos_items, neg_items = batch_data
        self.user_embeds, self.item_embeds = model.forward()

        self.constraint_mat = model.constraint_mat
        self.ii_neighbor_mat = model.ii_neighbor_mat
        self.ii_constraint_mat = model.ii_constraint_mat

        omega_weight = self.get_omegas(users, pos_items, neg_items)

        loss_L = self.cal_loss_L(users, pos_items, neg_items, omega_weight)
        loss_I = self.lambda_ * self.cal_loss_I(users, pos_items)
        norm_loss = self.gamma * self.norm_loss(model)
        
        loss = loss_L + loss_I + norm_loss
        losses = {
            'loss': loss,
            'loss_L': loss_L,
            'loss_I': loss_I,
            'norm_loss': norm_loss
        }
        return loss, losses
