import open_clip
import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import roc_curve as Roc
from scipy import interpolate
import numpy as np


def merge_yes_no_feature(dataset, model, device):
    txt = []
    N = len(dataset.classes)
    model.eval()
    if N:
        with open("./prompt/prompt.txt") as f:
            prompt_lis = f.readlines()
        num_prom = len(prompt_lis)
    for idx in range(num_prom):
        for name in dataset.classes:
            txt.append(open_clip.tokenize(prompt_lis[idx].replace("\n", "").format(name), 77).unsqueeze(0))
    txt = torch.cat(txt, dim=0)
    txt = txt.reshape(num_prom, len(dataset.classes), -1)
    text_inputs = txt.to(device)

    text_yes_ttl = torch.zeros(len(dataset.classes), 512).to(device)
    text_no_ttl = torch.zeros(len(dataset.classes), 512).to(device)

    with torch.no_grad():
        for i in range(num_prom):
            text_yes_i = model.encode_text(text_inputs[i])
            text_yes_i = F.normalize(text_yes_i, dim=-1)
            text_no_i = model.encode_text(text_inputs[i], "no")
            text_no_i = F.normalize(text_no_i, dim=-1)

            text_yes_ttl += text_yes_i
            text_no_ttl += text_no_i

    return F.normalize(text_yes_ttl, dim=-1), F.normalize(text_no_ttl, dim=-1)

def preprocess(batch):
    if isinstance(batch, dict):
        return batch

    if len(batch) == 2:
        batch = {'images': batch[0], 'labels': batch[1]}
    elif len(batch) == 3:
        batch = {'images': batch[0], 'labels': batch[1], 'metadata': batch[2]}
    else:
        raise ValueError(f'Unexpected number of elements: {len(batch)}')

    return batch

def cal_auc_fpr(ind_conf, ood_conf):
    conf = np.concatenate((ind_conf, ood_conf))
    ind_indicator = np.concatenate((np.ones_like(ind_conf), np.zeros_like(ood_conf)))
    auroc = metrics.roc_auc_score(ind_indicator, conf)
    fpr,tpr,thresh = Roc(ind_indicator, conf, pos_label=1)
    fpr = float(interpolate.interp1d(tpr, fpr)(0.95))
    return auroc, fpr