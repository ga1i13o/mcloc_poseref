import torch
from torch import nn
import numpy as np

from gloc.models.layers import L2Norm


class DenseFeaturesRefiner(nn.Module):
    def __init__(self, conf, ref_model):
        super().__init__() 
        self.conf = conf       
        self.ref_model = ref_model
        self.clamp  = conf.clamp
        self.norm = L2Norm()
        self.conf.bs = 32
        
    def forward(self, x):
        """
        Args:
            x (torch.tensor): batch of images shape Bx3xHxW
        Returns:
            torch.tensor: Features of shape BxDxHxW
        """
        with torch.no_grad():
            desc = self.ref_model(x)
            desc = self.norm(desc)

        return desc
    
    def score_candidates(self, q_feats, r_db_descriptors):
        """_summary_

        Args:
            q_feats (np.array): shape 1 x C x H x W
            r_db_descriptors (np.array): shape N_cand x C x H x W

        Returns:
            torch.tensor : vector of shape (N_cand, ), score of each one
        """
        q_feats = torch.tensor(q_feats)

        # this version is faster than looped, but requires much more memory due to broadcasting
        # r_db = torch.tensor(r_db_descriptors).squeeze(1)
        # scores = torch.linalg.norm(q_feats - r_db, dim=1) 
        scores = torch.zeros(len(r_db_descriptors), q_feats.shape[-2], q_feats.shape[-1])
        for i, desc in enumerate(r_db_descriptors):
            # q_feats : 1, D, H, W
            # desc    :    D, H, W
            # score   : 1, H, W
            score = torch.linalg.norm(q_feats - torch.tensor(desc), dim=1)
            scores[i] = score[0]

        if self.clamp > 0:
            scores = scores.clamp(max=self.clamp)
        scores = scores.sum(dim=(1,2)) / np.prod(scores.shape[-2:])

        return scores
        
    def rank_candidates(self, q_feats, r_db_descriptors, get_scores=False):
        scores = self.score_candidates(q_feats, r_db_descriptors)
        preds = torch.argsort(scores)
        if get_scores:
            return preds, scores[preds]
        return preds
