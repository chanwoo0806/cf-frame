import torch
import torch.nn.functional as F
from cf_frame.util import pload, pstore
from cf_frame.configurator import args


class NonParam:
    def __init__(self):
        self.type = 'nonparam'


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
    

# For LightGCL (ICLR 23)
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


class AFDLoss:
    def __init__(self):
        self.embed_reg = args.embed_reg
        self.alpha = args.alpha
        self.type = 'pairwise'

    def __call__(self, model, batch_data):
        ancs, poss, negs = batch_data

        # Forward pass to get embeddings
        user_embeds, item_embeds = model.forward()
        anc_embeds, pos_embeds, neg_embeds = user_embeds[ancs], item_embeds[poss], item_embeds[negs]

        # Compute BPR loss
        pos_preds = (anc_embeds * pos_embeds).sum(dim=-1)
        neg_preds = (anc_embeds * neg_embeds).sum(dim=-1)
        bpr_loss = torch.sum(torch.nn.functional.softplus(neg_preds - pos_preds)) / len(ancs)

        # Compute regularization loss
        anc_egos, pos_egos, neg_egos = model.user_embeds[ancs], model.item_embeds[poss], model.item_embeds[negs]
        reg_loss = (anc_egos.norm(2).pow(2) + pos_egos.norm(2).pow(2) + neg_egos.norm(2).pow(2)) / len(ancs)
        reg_loss *= self.embed_reg * 0.5

        # Compute correlation loss
        cor_loss_u, cor_loss_i = torch.zeros((1,)).to(args.device), torch.zeros((1,)).to(args.device)
        user_layer_correlations = []
        item_layer_correlations = []
        embeds_list = model.embeds_list
        for i in range(1, model.layer_num + 1):
            user_layer, item_layer = torch.split(embeds_list[i], [model.user_num, model.item_num])
            user_layer_correlations.append(self._calculate_correlation(user_layer))
            item_layer_correlations.append(self._calculate_correlation(item_layer))

        user_layer_correlations_coef = (1 / torch.tensor(user_layer_correlations)) / torch.sum(
            1 / torch.tensor(user_layer_correlations)
        )
        item_layer_correlations_coef = (1 / torch.tensor(item_layer_correlations)) / torch.sum(
            1 / torch.tensor(item_layer_correlations)
        )

        for i in range(1, model.layer_num + 1):
            cor_loss_u += user_layer_correlations_coef[i - 1] * user_layer_correlations[i - 1]
            cor_loss_i += item_layer_correlations_coef[i - 1] * item_layer_correlations[i - 1]

        cor_loss = self.alpha * (cor_loss_u + cor_loss_i)

        # Total loss
        loss = bpr_loss + reg_loss + cor_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'cor_loss': cor_loss}
        return loss, losses

    def _calculate_correlation(self, x):
        return x.T.corrcoef().triu(diagonal=1).norm()


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


# Infinite Layer Skip for UltraGCN
class LayerSkipLoss:
    def __init__(self):
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
        return loss.sum()
    
    def norm_loss(self, model):
        loss = 0.0
        for parameter in model.parameters():
            loss += torch.sum(parameter ** 2)
        return loss / 2
    
    def get_device(self):
        return args.device
    
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
    

# Cosine Contrastive Loss
class CCL:
    def __init__(self):
        self.neg_weight = args.neg_weight
        self.margin = args.margin
        self.embed_reg = args.embed_reg
        self.type = 'multineg_cpp'

    def _positive_loss(self, user_embeds, pos_embeds):
        pos_score = F.cosine_similarity(user_embeds, pos_embeds)
        pos_loss = (1 - pos_score).mean()
        return pos_loss

    def _negative_loss(self, user_embeds, neg_embeds):
        expanded_user_embeds = user_embeds.unsqueeze(-1).repeat(1, 1, args.negative_num)
        neg_scores = F.cosine_similarity(expanded_user_embeds, neg_embeds, dim=1)
        neg_loss = torch.clip(neg_scores - self.margin, min=0)
        neg_loss = (self.neg_weight / args.negative_num) * neg_loss.mean()
        return neg_loss
    
    def _l2_regularization(self, model):
        loss = 0.0
        for parameter in model.parameters():
            loss += torch.sum(parameter ** 2)
        return loss / 2

    def __call__(self, model, batch_data):
        ancs, poss, negs = batch_data
        user_embeds, item_embeds = model.forward()
        
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs].permute(dims=(0, 2, 1))  # Batch x Dimension x Negative

        pos_loss = self._positive_loss(anc_embeds, pos_embeds)
        neg_loss = self._negative_loss(anc_embeds, neg_embeds)
        reg_loss = self._l2_regularization(model)

        loss = pos_loss + neg_loss + self.embed_reg * reg_loss

        losses = {'ccl': loss, 'pos': pos_loss, 'neg': neg_loss, 'reg': reg_loss}        
        return loss, losses
    