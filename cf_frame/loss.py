import torch
import torch.nn.functional as F
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
    

# UltraGCN constraint loss
class Constraint:
    def __init__(self):
        self.weight_lambda = args.weight_lambda
        self.weight_gamma  = args.weight_gamma

    def _constraint_user_item(self, anc_embeds, pos_embeds, neg_embeds):
        # L_C loss
        pos_beta = 0
        neg_beta = 0

        pos_term = pos_beta * torch.log(F.sigmoid(anc_embeds.T * pos_embeds))
        neg_term = neg_beta * torch.log(F.sigmoid(-anc_embeds.T * neg_embeds))

        return pos_term + neg_term

    def _constraint_item_item(self, anc_embeds, pos_embeds):
        # L_I loss
        omega = 0

        topk_embeds = pass

        loss = omega * torch.log(F.sigmoid(anc_embeds.T * topk_embeds))
        return loss

    def _optimization(self, anc_embeds, pos_embeds, neg_embeds):
        # L_O loss
        pos_term = -torch.log(F.sigmoid(+anc_embeds.T * pos_embeds)).sum()
        neg_term = -torch.log(F.sigmoid(-anc_embeds.T * neg_embeds)).sum()
        return pos_term + neg_term

    def __call__(self, model, batch_data):
        ancs, poss, negs = batch_data
        user_embeds, item_embeds = model.forward()

        anc_embeds, pos_embeds, neg_embeds = user_embeds[ancs], item_embeds[poss], item_embeds[negs]

        constraint_user_item_loss = self._constraint_user_item(anc_embeds, pos_embeds, neg_embeds)
        constraint_item_item_loss = self._constraint_item_item(anc_embeds, pos_embeds)
        optimization_loss = self._optimization(anc_embeds, pos_embeds, neg_embeds)

        loss = constraint_item_item_loss + constraint_user_item_loss + optimization_loss
        losses = {
            'user_item': constraint_user_item_loss,
            'item_item': constraint_item_item_loss,
            'optimization': optimization_loss
        }
        return loss, losses
