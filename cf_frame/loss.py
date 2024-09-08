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
    

# Cosine Contrastive Loss
class CCL:
    def __init__(self):
        self.neg_weight = args.neg_weight
        self.margin = args.margin
        self.embed_reg = args.embed_reg
        self.type = 'multineg'

    def _positive_loss(self, user_embeds, pos_embeds):
        pos_score = F.cosine_similarity(user_embeds, pos_embeds)
        pos_loss = (1 - pos_score).mean()
        return pos_loss

    def _negative_loss(self, user_embeds, neg_embeds):
        neg_loss = 0
        neg_num = args.neg_num
        for i in range(neg_num):
            neg_score = F.cosine_similarity(user_embeds, neg_embeds[:, :, i])
            neg_loss += torch.clip(neg_score - self.margin, min=0)
        neg_loss = (self.neg_weight / neg_num) * neg_loss.mean()
        return neg_loss
    
    def _l2_regularization(self, *embeds):
        l2_loss = 0
        for embed in embeds:
            l2_loss += torch.sum(embed.pow(2))
        return l2_loss

    def __call__(self, model, batch_data):
        ancs, poss, negs = batch_data
        user_embeds, item_embeds = model.forward()
        
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs].permute(dims=(0, 2, 1))  # Batch x Dimension x Negative

        pos_loss = self._positive_loss(anc_embeds, pos_embeds)
        neg_loss = self._negative_loss(anc_embeds, neg_embeds)
        reg_loss = self._l2_regularization(anc_embeds, pos_embeds, neg_embeds)

        loss = pos_loss + neg_loss + self.embed_reg * reg_loss
        loss.requires_grad = True

        losses = {'ccl': loss, 'pos': pos_loss, 'neg': neg_loss, 'reg': reg_loss}        
        return loss, losses