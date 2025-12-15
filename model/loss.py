import torch.nn.functional as F

def l1_loss(output, target):
    return F.l1_loss(output, target)



def nll_loss(output, target):
    return F.nll_loss(output, target)
