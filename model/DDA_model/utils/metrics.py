"""
Code is adapted from https://github.com/SebastianHafner/DDA_UrbanExtraction
Modified: Arno Rüegg, Nando Metzger
"""
import torch
import numpy as np
import sys
import kornia
from scipy.spatial.distance import directed_hausdorff


class MultiThresholdMetric(object):
    def __init__(self, threshold):

        self._thresholds = threshold[ :, None, None, None, None] # [Tresh, B, C, H, W]
        self._data_dims = (-1, -2, -3, -4) # For a B/W image, it should be [Thresh, B, C, H, W],

        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

    def add_sample(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        y_true = y_true.bool()[None,...] # [Thresh, B,  C, ...]
        y_pred = y_pred[None, ...]  # [Thresh, B, C, ...]
        y_pred_offset = (y_pred - self._thresholds + 0.5).round().bool()

        self.TP += (y_true & y_pred_offset).sum(dim=self._data_dims).float()
        self.TN += (~y_true & ~y_pred_offset).sum(dim=self._data_dims).float()
        self.FP += (y_true & ~y_pred_offset).sum(dim=self._data_dims).float()
        self.FN += (~y_true & y_pred_offset).sum(dim=self._data_dims).float()

    @property
    def precision(self):
        if hasattr(self, '_precision'):
            '''precision previously computed'''
            return self._precision

        denom = (self.TP + self.FP).clamp(10e-05)
        self._precision = self.TP / denom
        return self._precision

    @property
    def recall(self):
        if hasattr(self, '_recall'):
            '''recall previously computed'''
            return self._recall

        denom = (self.TP + self.FN).clamp(10e-05)
        self._recall = self.TP / denom
        return self._recall
    
    @property
    def IoU(self):
        if hasattr(self, '_IoU'):
            '''IoU previously computed'''
            return self._IoU

        denom = (self.TP + self.FP + self.FN).clamp(10e-05)
        self._IoU = self.TP / denom
        return self._IoU


    def compute_basic_metrics(self):
        '''
        Computes False Negative Rate and False Positive rate
        :return:
        '''

        false_pos_rate = self.FP/(self.FP + self.TN)
        false_neg_rate = self.FN / (self.FN + self.TP)

        return false_pos_rate, false_neg_rate

    def compute_f1(self):
        denom = (self.precision + self.recall).clamp(10e-05)
        return 2 * self.precision * self.recall / denom


def true_pos(y_true: torch.Tensor, y_pred: torch.Tensor, dim=0):
    return torch.sum(y_true * torch.round(y_pred), dim=dim)  # Only sum along H, W axis, assuming no C


def false_pos(y_true, y_pred, dim=0):
    return torch.sum((1. - y_true) * torch.round(y_pred), dim=dim)


def false_neg(y_true: torch.Tensor, y_pred: torch.Tensor, dim=0):
    return torch.sum(y_true * (1. - torch.round(y_pred)), dim=dim)


def precision(y_true: torch.Tensor, y_pred: torch.Tensor, dim: int):
    TP = true_pos(y_true, y_pred, dim)
    FP = false_pos(y_true, y_pred, dim)
    denom = TP + FP
    denom = torch.clamp(denom, 10e-05)
    return TP / denom


def IoU(y_true: torch.Tensor, y_pred: torch.Tensor, dim: int):
    TP = true_pos(y_true, y_pred, dim)
    FN = false_neg(y_true, y_pred, dim)
    FP = false_pos(y_true, y_pred, dim)
    denom = TP + FP + FN
    denom = torch.clamp(denom, 10e-05)
    return TP / denom

def boundary_IoU(y_true: torch.Tensor, y_pred: torch.Tensor):
    gt_boundary = kornia.morphology.dilation(y_true, torch.ones((3,3)).cuda())
    pred_boundary = kornia.morphology.dilation(y_pred, torch.ones((3,3)).cuda())
    IoU_boundary = IoU(gt_boundary, pred_boundary,dim=None)
    return IoU_boundary

def hausdorff(gt: torch.Tensor, pred: torch.Tensor):
    batch_size = gt.size(0)
    hausdorff_distances = []

    for i in range(batch_size):
        gt_mask = gt[i].squeeze().cpu().numpy()
        pred_mask = pred[i].squeeze().cpu().numpy()

        gt_coords = np.array(list(zip(*np.nonzero(gt_mask))))
        pred_coords = np.array(list(zip(*np.nonzero(pred_mask))))

        hausdorff_distance = directed_hausdorff(gt_coords, pred_coords)[0]
        hausdorff_distances.append(hausdorff_distance)

    hausdorff_distances = torch.tensor(hausdorff_distances)

    return hausdorff_distances


def ssim(y_true: torch.Tensor, y_pred: torch.Tensor, windowsize=7):
    ssim = kornia.metrics.ssim(y_true,y_pred,windowsize)
    return torch.mean(ssim)

def closed_IoU(y_true: torch.Tensor, y_pred: torch.Tensor):
    gt = kornia.morphology.closing(y_true, torch.ones((3,3)).cuda())
    pred = kornia.morphology.closing(y_pred, torch.ones((3,3)).cuda())
    return IoU(gt, pred,dim=None)

def opened_IoU(y_true: torch.Tensor, y_pred: torch.Tensor):
    gt = kornia.morphology.opening(y_true, torch.ones((3,3)).cuda())
    pred = kornia.morphology.opening(y_pred, torch.ones((3,3)).cuda())
    return IoU(gt, pred,dim=None)

def gradient_IoU(y_true: torch.Tensor, y_pred: torch.Tensor):
    kernel = torch.Tensor([[0,-1,0],[-1,4,-1],[0,-1,0]]).cuda()
    gt_grad = kornia.morphology.gradient(y_true, kernel)
    pred_grad = kornia.morphology.gradient(y_pred, kernel)
        
    gt = kornia.morphology.dilation(gt_grad, torch.ones((3,3)).cuda())
    pred = kornia.morphology.dilation(pred_grad, torch.ones((3,3)).cuda())

    return IoU(gt, pred,dim=None)

def recall(y_true: torch.Tensor, y_pred: torch.Tensor, dim: int):
    TP = true_pos(y_true, y_pred, dim)
    FN = false_neg(y_true, y_pred, dim)
    denom = TP + FN
    denom = torch.clamp(denom, 10e-05)
    return true_pos(y_true, y_pred, dim) / denom


def f1_score(gts:torch.Tensor, preds:torch.Tensor, multi_threashold_mode=False, dim=(-1, -2)):
    # FIXME Does not operate proper
    gts = gts.float()
    preds = preds.float()

    if multi_threashold_mode:
        gts = gts[:, None, ...] # [B, Thresh, ...]
        gts = gts.expand_as(preds)

    with torch.no_grad():
        recall_val = recall(gts, preds, dim)
        precision_val = precision(gts, preds, dim)
        denom = torch.clamp( (recall_val + precision_val), 10e-5)

        f1 = 2. * recall_val * precision_val / denom

    return f1


def f1_score_from_prob(y_prob: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    p = precision_from_prob(y_prob, y_true, threshold=threshold)
    r = recall_from_prob(y_prob, y_true, threshold=threshold)
    return 2 * (p * r) / (p + r + sys.float_info.epsilon)


def true_positives_from_prob(y_prob: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    y_pred = y_prob > threshold
    tp = np.sum(np.logical_and(y_pred, y_true))
    return tp.astype(np.int64)


def true_negatives_from_prob(y_prob: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    y_pred = y_prob > threshold
    tn = np.sum(np.logical_and(np.logical_not(y_pred), np.logical_not(y_true)))
    return tn.astype(np.int64)


def false_positives_from_prob(y_prob: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    y_pred = y_prob > threshold
    fp = np.sum(np.logical_and(y_pred, np.logical_not(y_true)))
    return fp.astype(np.int64)


def false_negatives_from_prob(y_prob: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    y_pred = y_prob > threshold
    fn = np.sum(np.logical_and(np.logical_not(y_pred), y_true))
    return fn.astype(np.int64)


def precision_from_prob(y_prob: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    tp = true_positives_from_prob(y_prob, y_true, threshold)
    fp = false_positives_from_prob(y_prob, y_true, threshold)
    return tp / (tp + fp)


def recall_from_prob(y_prob: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    tp = true_positives_from_prob(y_prob, y_true, threshold)
    fn = false_negatives_from_prob(y_prob, y_true, threshold)
    return tp / (tp + fn + sys.float_info.epsilon)


def iou_from_prob(y_prob: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    tp = true_positives_from_prob(y_prob, y_true, threshold)
    fp = false_positives_from_prob(y_prob, y_true, threshold)
    fn = false_negatives_from_prob(y_prob, y_true, threshold)
    return tp / (tp + fp + fn)


def kappa_from_prob(y_prob: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    tp = true_positives_from_prob(y_prob, y_true, threshold)
    tn = true_negatives_from_prob(y_prob, y_true, threshold)
    fp = false_positives_from_prob(y_prob, y_true, threshold)
    fn = false_negatives_from_prob(y_prob, y_true, threshold)
    nominator = 2 * (tp * tn - fn * fp)
    denominator = (tp + fp) * (fp + tn) + (tp + fn) * (fn + tn)
    return nominator / denominator


def root_mean_square_error(y_pred: np.ndarray, y_true: np.ndarray):
    return np.sqrt(np.sum(np.square(y_pred - y_true)) / np.size(y_true))
