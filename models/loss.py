import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class BCELoss(nn.Module):
    def __init__(self, ignore_index=255, ignore_bg=True, pos_weight=None, reduction='mean'):
        super().__init__()
        self.ignore_index = ignore_index
        self.pos_weight = pos_weight
        self.reduction = reduction

        if ignore_bg is True:
            self.ignore_indexes = [0, self.ignore_index]
        else:
            self.ignore_indexes = [self.ignore_index]

    def forward(self, logit, label, logit_old=None):
        # logit:     [N, C_tot, H, W]
        # logit_old: [N, C_prev, H, W]
        # label:     [N, H, W] or [N, C, H, W]
        C = logit.shape[1]
        if logit_old is None:
            if len(label.shape) == 3:
                # target: [N, C, H, W]
                target = torch.zeros_like(logit).float().to(logit.device)
                for cls_idx in label.unique():
                    if cls_idx in self.ignore_indexes:
                        continue
                    target[:, int(cls_idx)] = (label == int(cls_idx)).float()
            elif len(label.shape) == 4:
                target = label
            else:
                raise NotImplementedError
            
            logit = logit.permute(0, 2, 3, 1).reshape(-1, C)
            target = target.permute(0, 2, 3, 1).reshape(-1, C)

            return nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction=self.reduction)(logit, target)
        else:
            if len(label.shape) == 3:
                # target: [N, C, H, W]
                target = torch.zeros_like(logit).float().to(logit.device)
                target[:, 1:logit_old.shape[1]] = logit_old.sigmoid()[:, 1:]
                for cls_idx in label.unique():
                    if cls_idx in self.ignore_indexes:
                        continue
                    target[:, int(cls_idx)] = (label == int(cls_idx)).float()
            else:
                raise NotImplementedError
            
            loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction=self.reduction)(logit, target)
            del target

            return loss


class WBCELoss(nn.Module):
    def __init__(self, ignore_index=255, pos_weight=None, reduction='none', n_old_classes=0, n_new_classes=0):
        super().__init__()
        self.ignore_index = ignore_index
        self.n_old_classes = n_old_classes  # |C0:t-1| + 1(bg), 19-1: 20 | 15-5: 16 | 15-1: 16...
        self.n_new_classes = n_new_classes  # |Ct|, 19-1: 1 | 15-5: 5 | 15-1: 1

        self.reduction = reduction
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=self.reduction)

    def forward(self, logit, label):
        # logit:     [N, |Ct|, H, W]
        # label:     [N, H, W]

        N, C, H, W = logit.shape
        target = torch.zeros_like(logit, device=logit.device).float()
        for cls_idx in label.unique():
            if cls_idx in [0, self.ignore_index]:
                continue
            target[:, int(cls_idx) - self.n_old_classes] = (label == int(cls_idx)).float()


        loss = self.criterion(
            logit.permute(0, 2, 3, 1).reshape(-1, C),
            target.permute(0, 2, 3, 1).reshape(-1, C)
        )

        if self.reduction == 'none':
            return loss.reshape(N, H, W, C).permute(0, 3, 1, 2)  # [N, C, H, W]
        elif self.reduction == 'mean':
            return loss
        else:
            raise NotImplementedError




class KDLoss(nn.Module):
    def __init__(self, pos_weight=None, reduction='mean'):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)

    def forward(self, logit, logit_old=None):
        # logit:     [N, |Ct|, H, W]
        # logit_old: [N, |Ct|, H, W]
        
        N, C, H, W = logit.shape
        loss = self.criterion(
            logit.permute(0, 2, 3, 1).reshape(-1, C),
            logit_old.permute(0, 2, 3, 1).reshape(-1, C)
        ).reshape(N, H, W, C).permute(0, 3, 1, 2)
        return loss


#  辅助分类器
class ACLoss(nn.Module):
    def __init__(self, reduction='none'):
        super().__init__()
        self.reduction = reduction
        self.criterion = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, logit):
        # logit: [N, 1, H, W]
        
        return self.criterion(logit, torch.zeros_like(logit))
        # loss = -torch.log(1 - logit.sigmoid())



class UnbiasedKnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, mask=None):   

        new_cl = inputs.shape[1] - targets.shape[1]   

        targets = targets * self.alpha                

        new_bkg_idx = torch.tensor([0] + [x for x in range(targets.shape[1], inputs.shape[1])]).to(inputs.device)   

        den = torch.logsumexp(inputs, dim=1)                          # B, H, W
        outputs_no_bgk = inputs[:, 1:-new_cl] - den.unsqueeze(dim=1)  # B, OLD_CL, H, W
        outputs_bkg = torch.logsumexp(torch.index_select(inputs, index=new_bkg_idx, dim=1), dim=1) - den    

        labels = torch.softmax(targets, dim=1)                        # B, BKG + OLD_CL, H, W

        # make the average on the classes 1/n_cl \sum{c=1..n_cl} L_c
        loss = (labels[:, 0] * outputs_bkg + (labels[:, 1:] * outputs_no_bgk).sum(dim=1)) / targets.shape[1]    

        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
                outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
                outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs



class UnbiasedCrossEntropy(nn.Module):
    def __init__(self, old_cl=None, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.old_cl = old_cl

    def forward(self, inputs, targets):    
        old_cl = self.old_cl
        outputs = torch.zeros_like(inputs)  # B, C (1+V+N), H, W
        den = torch.logsumexp(inputs, dim=1)                              
        outputs[:, 0] = torch.logsumexp(inputs[:, 0:old_cl], dim=1) - den    
        outputs[:, old_cl:] = inputs[:, old_cl:] - den.unsqueeze(dim=1)    

        labels = targets.clone()    # B, H, W
        labels[targets < old_cl] = 0  # just to be sure that all labels old belongs to zero

        b=outputs.type()
        labels = labels.long()
        a=labels.type()
        loss = F.nll_loss(outputs, labels, ignore_index=self.ignore_index, reduction=self.reduction)

        return loss