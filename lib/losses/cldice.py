import torch
import torch.nn as nn
import torch.nn.functional as F
from .soft_skeleton import soft_skel

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class soft_cldice(nn.Module):
    def __init__(self, iter_=3, smooth = 1.):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth

    def forward(self,y_true, y_pred):
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:,1:,...])+self.smooth)/(torch.sum(skel_pred[:,1:,...])+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:,1:,...])+self.smooth)/(torch.sum(skel_true[:,1:,...])+self.smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice

def soft_dice(y_true, y_pred):
    """[function to compute dice loss]

    Args:
        y_true ([float32]): [ground truth image]
        y_pred ([float32]): [predicted image]

    Returns:
        [float32]: [loss value]
    """
    smooth = 1
    num=y_pred.size(0);
    y_true=y_true.view(num,-1);
    y_pred=y_pred.view(num,-1);
    #intersection = torch.sum((y_true * y_pred)[:,1:,...])
    intersection = torch.sum((y_true * y_pred))
    coeff = (2. *  intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)
    return (1. - coeff)

class WeightedBCE(nn.Module):

    def __init__(self, weights=[0.4, 0.6]):
        super(WeightedBCE, self).__init__()
        self.weights = weights

    def forward(self, logit_pixel, truth_pixel):
        # print("====",logit_pixel.size())
        logit = logit_pixel.view(-1)
        truth = truth_pixel.view(-1)
        assert(logit.shape==truth.shape)
        loss = F.binary_cross_entropy(logit, truth, reduction='none')
        pos = (truth>0.5).float()
        neg = (truth<0.5).float()
        pos_weight = pos.sum().item() + 1e-12
        neg_weight = neg.sum().item() + 1e-12
        loss = (self.weights[0]*pos*loss/pos_weight + self.weights[1]*neg*loss/neg_weight).sum()

        return loss
class soft_dice_cldice(nn.Module):    
    def __init__(self, iter_=3, alpha=0.5, smooth = 1.):
        #super(soft_cldice, self).__init__()
        super(soft_dice_cldice, self).__init__()
        self.BCE_loss = WeightedBCE(weights=[0.5, 0.5])
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha
        self.fcoal_loss=FocalLoss();
    def forward(self, y_pred,y_true):
        
        dice = soft_dice(y_true, y_pred)
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        num=y_pred.size(0);
        y_true=y_true.view(num,-1);
        y_pred=y_pred.view(num,-1);
        skel_pred=skel_pred.view(num,-1);
        skel_true=skel_true.view(num,-1);
        #tprec = (torch.sum(torch.mul(skel_pred, y_true)[:,1:,...])+self.smooth)/(torch.sum(skel_pred[:,1:,...])+self.smooth)    
        #tsens = (torch.sum(torch.mul(skel_true, y_pred)[:,1:,...])+self.smooth)/(torch.sum(skel_true[:,1:,...])+self.smooth)    
        tprec = (torch.sum(torch.mul(skel_pred, y_true))+self.smooth)/(torch.sum(skel_pred)+self.smooth)    
        tsens = (torch.sum(torch.mul(skel_true, y_pred))+self.smooth)/(torch.sum(skel_true)+self.smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        #print("loss",(1.0-self.alpha)*dice+self.alpha*cl_dice);
        cldicex=(1.0-self.alpha)*dice+self.alpha*cl_dice;
        #cldicex=0.7*self.fcoal_loss(y_pred,y_true)+0.3*cldicex;
        bce=self.BCE_loss(y_pred,y_true);
        return 0.3*cldicex+0.7*bce;
        #cldicex=self.fcoal_loss(y_pred,y_true)
        #return cldicex

class  Poly_focal_loss(nn.Module):
    def __init__(self, epsilon=1.0,alpha=1, gamma=2, logits=False, reduce=True):
        super(Poly_focal_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.epsilon=epsilon
    def forward(self, inputs, targets):
        
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        pt1=BCE_loss*inputs+(1-targets)*(1-inputs);
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduce:
            F_loss=torch.mean(F_loss)
        #2022 5.6
        poly1=F_loss+ self.epsilon*torch.pow(1-pt1,self.gamma+1)
        return poly1
class  poly1_cross_entropy(nn.Module):
     def __init__(self,epsilon=1.0):
        super( poly1_cross_entropy, self).__init__()
        self.epsilon=epsilon
        self.CE_loss=torch.nn.CrossEntropyLoss();
     def forward(self, y_pred,y_true):
        pt=torch.sum(y_pred*y_true,dim=-1);
        CE=self.CE_loss(y_pred,y_true);
        Poly1=CE+self.epsilon*(1-pt);
        print("poly:",Poly1);
        return Poly1;
        