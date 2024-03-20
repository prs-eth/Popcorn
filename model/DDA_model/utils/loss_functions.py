"""
Code is adapted from https://github.com/SebastianHafner/DDA_UrbanExtraction
Modified: Arno Rüegg, Nando Metzger
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


def get_criterion(loss_type, negative_weight: float = 1, positive_weight: float = 1):

    if loss_type == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    elif loss_type == 'CrossEntropyLoss':
        balance_weight = [negative_weight, positive_weight]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        balance_weight = torch.tensor(balance_weight).float().to(device)
        criterion = nn.CrossEntropyLoss(weight=balance_weight)
    elif loss_type == 'SoftDiceLoss':
        criterion = soft_dice_loss
    elif loss_type == 'SoftDiceSquaredSumLoss':
        criterion = soft_dice_squared_sum_loss
    elif loss_type == 'SoftDiceBalancedLoss':
        criterion = soft_dice_loss_balanced
    elif loss_type == 'PowerJaccardLoss':
        criterion = power_jaccard_loss
    elif loss_type == 'MeanSquareErrorLoss':
        criterion = nn.MSELoss()
    elif loss_type == 'IoULoss':
        criterion = iou_loss
    elif loss_type == 'DiceLikeLoss':
        criterion = dice_like_loss
    else:
        raise Exception(f'unknown loss {loss_type}')

    return criterion


def soft_dice_loss(y_logit: torch.Tensor, y_true: torch.Tensor):
    y_prob = torch.sigmoid(y_logit)
    eps = 1e-6

    y_prob = y_prob.flatten()
    y_true = y_true.flatten()
    intersection = (y_prob * y_true).sum()

    return 1 - ((2. * intersection + eps) / (y_prob.sum() + y_true.sum() + eps))


# TODO: fix this one
def soft_dice_squared_sum_loss(y_logit: torch.Tensor, y_true: torch.Tensor):
    y_prob = torch.sigmoid(y_logit)
    eps = 1e-6

    y_prob = y_prob.flatten()
    y_true = y_true.flatten()
    intersection = (y_prob * y_true).sum()

    return 1 - ((2. * intersection + eps) / (y_prob.sum() + y_true.sum() + eps))


def soft_dice_loss_multi_class(input:torch.Tensor, y:torch.Tensor):
    p = torch.softmax(input, dim=1)
    eps = 1e-6

    sum_dims= (0, 2, 3) # Batch, height, width

    intersection = (y * p).sum(dim=sum_dims)
    denom =  (y.sum(dim=sum_dims) + p.sum(dim=sum_dims)).clamp(eps)

    loss = 1 - (2. * intersection / denom).mean()
    return loss


def soft_dice_loss_multi_class_debug(input:torch.Tensor, y:torch.Tensor):
    p = torch.softmax(input, dim=1)
    eps = 1e-6

    sum_dims= (0, 2, 3) # Batch, height, width

    intersection = (y * p).sum(dim=sum_dims)
    denom =  (y.sum(dim=sum_dims) + p.sum(dim=sum_dims)).clamp(eps)

    loss = 1 - (2. * intersection / denom).mean()
    loss_components = 1 - 2 * intersection/denom
    return loss, loss_components


def generalized_soft_dice_loss_multi_class(input:torch.Tensor, y:torch.Tensor):
    p = torch.softmax(input, dim=1)
    eps = 1e-12

    # TODO [B, C, H, W] -> [C, B, H, W] because softdice includes all pixels

    sum_dims= (0, 2, 3) # Batch, height, width
    ysum = y.sum(dim=sum_dims)
    wc = 1 / (ysum ** 2 + eps)
    intersection = ((y * p).sum(dim=sum_dims) * wc).sum()
    denom =  ((ysum + p.sum(dim=sum_dims)) * wc).sum()

    loss = 1 - (2. * intersection / denom)
    return loss


def jaccard_like_loss_multi_class(input:torch.Tensor, y:torch.Tensor):
    p = torch.softmax(input, dim=1)
    eps = 1e-6

    # TODO [B, C, H, W] -> [C, B, H, W] because softdice includes all pixels

    sum_dims= (0, 2, 3) # Batch, height, width

    intersection = (y * p).sum(dim=sum_dims)
    denom =  (y ** 2 + p ** 2).sum(dim=sum_dims) + (y*p).sum(dim=sum_dims) + eps

    loss = 1 - (2. * intersection / denom).mean()
    return loss


def jaccard_like_loss(input:torch.Tensor, target:torch.Tensor):
    input_sigmoid = torch.sigmoid(input)
    eps = 1e-6

    iflat = input_sigmoid.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()
    denom = (iflat**2 + tflat**2).sum() - (iflat * tflat).sum() + eps

    return 1 - ((2. * intersection) / denom)


def dice_like_loss(input: torch.Tensor, target: torch.Tensor):
    input_sigmoid = torch.sigmoid(input)
    eps = 1e-6

    iflat = input_sigmoid.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()
    denom = (iflat**2 + tflat**2).sum() + eps

    return 1 - ((2. * intersection) / denom)


def power_jaccard_loss(input: torch.Tensor, target: torch.Tensor):
    input_sigmoid = torch.sigmoid(input)
    eps = 1e-6

    iflat = input_sigmoid.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()
    denom = (iflat**2 + tflat**2).sum() - (iflat * tflat).sum() + eps

    return 1 - (intersection / denom)


def iou_loss(y_logit: torch.Tensor, y_true: torch.Tensor):
    y_pred = torch.sigmoid(y_logit)
    eps = 1e-6

    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    intersection = (y_pred * y_true).sum()
    union = (y_pred + y_true).sum() - intersection + eps

    return 1 - (intersection / union)


def jaccard_like_balanced_loss(input:torch.Tensor, target:torch.Tensor):
    input_sigmoid = torch.sigmoid(input)
    eps = 1e-6

    iflat = input_sigmoid.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()
    denom = (iflat**2 + tflat**2).sum() - (iflat * tflat).sum() + eps
    piccard = (2. * intersection)/denom

    n_iflat = 1-iflat
    n_tflat = 1-tflat
    neg_intersection = (n_iflat * n_tflat).sum()
    neg_denom = (n_iflat**2 + n_tflat**2).sum() - (n_iflat * n_tflat).sum()
    n_piccard = (2. * neg_intersection)/neg_denom

    return 1 - piccard - n_piccard


def soft_dice_loss_balanced(input:torch.Tensor, target:torch.Tensor):
    input_sigmoid = torch.sigmoid(input)
    eps = 1e-6

    iflat = input_sigmoid.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()

    dice_pos = ((2. * intersection) /
                (iflat.sum() + tflat.sum() + eps))

    negatiev_intersection = ((1-iflat) * (1 - tflat)).sum()
    dice_neg =  (2 * negatiev_intersection) / ((1-iflat).sum() + (1-tflat).sum() + eps)

    return 1 - dice_pos - dice_neg