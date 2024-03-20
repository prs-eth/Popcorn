"""
Project: 🍿POPCORN: High-resolution Population Maps Derived from Sentinel-1 and Sentinel-2 🌍🛰️
Nando Metzger, 2024
"""

import torch

from utils.losses import r2, mape_func
import torch.nn.functional as F


def get_test_metrics(pred, y, tag=""):

    log_dict = {
        "l1_loss":  F.l1_loss(pred, y),
        "r2": r2(pred, y),
        "mape": mape_func(pred, y),
        "log_l1_loss": F.l1_loss(torch.log(pred+1), torch.log(y+1)),
        "mse_loss": F.mse_loss(pred, y),
        "log_mse_loss": F.mse_loss(torch.log(pred+1), torch.log(y+1)),
        "Correlation": torch.corrcoef(torch.stack([pred, y]))[0,1]
    }
    log_dict = {"Population_" + tag + "/"+key: value for key,value in log_dict.items()}

    return log_dict