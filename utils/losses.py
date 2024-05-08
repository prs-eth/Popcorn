"""
Project: 🍿POPCORN: High-resolution Population Maps Derived from Sentinel-1 and Sentinel-2 🌍🛰️
Nando Metzger, 2024
"""

import torch.nn.functional as F
import torch

from collections import defaultdict


def get_loss(output, gt, scale=None,
             loss=["l1_loss"], lam=[1.0],
             tag="",
             scale_regularization=0.0,
             ):
    """
    Compute the loss for the model
    input:
        output: dict of model outputs
        gt: dict of ground truth
        scale: tensor, the scale e.g. the occupancy map
        tag: str, tag to be used for logging
        scale_regularization: float, weight for the scale regularization
    output:
        loss: float, the loss
        auxdict: dict, auxiliary losses
    """
    auxdict = defaultdict(float)

    # check that all tensors are float32
    if output["popcount"].dtype != torch.float32:
        output["popcount"] = output["popcount"].float()
    
    if output["popdensemap"].dtype != torch.float32:
        output["popdensemap"] = output["popdensemap"].float()

    if output["scale"] is not None:
        if output["scale"].dtype != torch.float32:
            output["scale"] = output["scale"].float()
    
    # prepare vars1.0
    y_pred = output["popcount"]
    y_gt = gt["y"]
    if "popvar" in output.keys():
        if output["popvar"].dtype != torch.float32:
            output["popvar"] = output["popvar"].float()
        var = output["popvar"]

    # Population loss and metrics
    metricdict = {
        "l1_loss": F.l1_loss(y_pred, y_gt),
        "log_l1_loss": F.l1_loss(torch.log(y_pred+1), torch.log(y_gt+1)),
        "mse_loss": F.mse_loss(y_pred, y_gt),
        "log_mse_loss": F.mse_loss(torch.log(y_pred+1), torch.log(y_gt+1)),
        "mr2": r2(y_pred, y_gt) if len(y_pred)>1 else torch.tensor(0.0),
        "mape": mape_func(y_pred, y_gt),
        "mCorrelation": torch.corrcoef(torch.stack([y_pred, y_gt]))[0,1] if len(y_pred)>1 else torch.tensor(0.0),
    }

    # define optimization loss as a weighted sum of the losses
    optimization_loss = torch.tensor(0, device=y_pred.device, dtype=y_pred.dtype)
    for lo,la in zip(loss,lam):
        if lo in metricdict.keys():
            optimization_loss += metricdict[lo] * la

    # occupancy scale regularization
    if scale is not None:
        if torch.isnan(scale).any():
            raise ValueError("nan values detected in scale.")
        if torch.isinf(scale).any():
            raise ValueError("inf values detected in scale.")
        
        metricdict["scale"] = scale.float().abs().mean()
        if scale_regularization>0.0:
            optimization_loss += scale_regularization * metricdict["scale"]

    # prepare for logging
    if tag=="":
        auxdict = {**auxdict, **{"Population"+"/"+key: value for key,value in metricdict.items()}}
    else:
        auxdict = {**auxdict, **{"Population_"+tag+"/"+key: value for key,value in metricdict.items()}}

    # prepare for logging
    auxdict["optimization_loss"] =  optimization_loss
    auxdict = {key:value.detach().item() for key,value in auxdict.items()}

    return optimization_loss, auxdict


def mape_func(pred, gt, eps=1e-8):
    """
    Calculate the mean absolute percentage error between the ground truth and the prediction.
    """
    pos_mask = gt>0.1
    mre =  ( (pred[pos_mask]- gt[pos_mask]).abs() / (gt[pos_mask] + eps)).mean()
    return mre*100


# adapted from https://stackoverflow.com/questions/65840698/how-to-make-r2-score-in-nn-lstm-pytorch
def r2(pred, gt, eps=1e-8):
    """
    Calculate the R2 score between the ground truth and the prediction.
    
    Parameters
    ----------
    pred : tensor
        The predicted values.
    gt : tensor
        Ground truth values.

    Returns
    -------
    r2 : tensor
        The R2 score.

    Forumula
    --------
    R2 = 1 - SS_res / SS_tot
    SS_res = sum((gt - pred) ** 2)
    SS_tot = sum((gt - gt_mean) ** 2)
    """
    gt_mean = torch.mean(gt)
    ss_tot = torch.sum((gt - gt_mean) ** 2)
    ss_res = torch.sum((gt - pred) ** 2)
    r2 = 1 - ss_res / (ss_tot + eps)
    return r2
