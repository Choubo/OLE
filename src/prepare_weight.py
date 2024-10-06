import os
import random

import torch
import numpy as np
import open_clip
import torch.nn.functional as F
import pickle
from sklearn import mixture
from utils.cfg import parse_arguments
from util import merge_yes_no_feature
from utils.datasets import ImageNet

def encoder_outlier_prompt(model, device, label_path):
    with open(label_path, 'rb') as f:
        outlier_label = np.load(f)
    txt = []
    N = len(outlier_label)
    model.eval()
    if N:
        with open("./prompt/prompt.txt") as f:
            prompt_lis = f.readlines()
        num_prom = len(prompt_lis)
    for name in outlier_label:
        for idx in range(num_prom):
            txt.append(open_clip.tokenize(prompt_lis[idx].replace("\n", "").format(name), 77).unsqueeze(0))
    txt = torch.cat(txt, dim=0)
    txt = txt.reshape(len(outlier_label), num_prom, -1)
    text_inputs = txt.to(device)

    text_ttl = []

    with torch.no_grad():
        for i in range(N):
            text_i = model.encode_text(text_inputs[i])
            text_i = F.normalize(text_i, dim=-1)
            text_i = text_i.mean(dim=0)
            text_i = F.normalize(text_i, dim=-1)
            text_ttl.append(text_i)

    return torch.stack(text_ttl, dim=1).T

def prototype_learning(weight, n_prototypes):
    cluster = mixture.GaussianMixture(n_components=n_prototypes, random_state=42).fit(weight.cpu().numpy())
    centroid = cluster.means_
    centroid = torch.from_numpy(centroid).cuda()
    centroid /= centroid.norm(dim=-1, keepdim=True)
    return centroid.float()


def refinement(weight_yes, outlier_weight, quantile=1):
    to_np = lambda x: x.data.cpu().numpy()
    output = outlier_weight @ weight_yes.T
    output = np.max(to_np(output), axis=1)
    outlier_weight = outlier_weight[output<np.quantile(output, quantile)]
    return outlier_weight, np.quantile(output, quantile)

def ID_Cluster(weight):
    n_clusters = 5
    cluster = mixture.GaussianMixture(n_components=n_clusters, random_state=42).fit(weight.cpu().numpy())
    centroid = cluster.means_
    centroid = torch.from_numpy(centroid).cuda().float()
    centroid /= centroid.norm(dim=-1, keepdim=True)
    labels = cluster.predict(weight.cpu().numpy())
    datalist = []
    for i in range(n_clusters):
        datalist.append(weight[labels == i])

    far_samples = []
    for i, (c, data) in enumerate(zip(centroid, datalist)):
        distance = data @ c
        k_th_distance, minD_idx = torch.topk(distance, 30, largest=False)
        far_samples.append(data[minD_idx])

    far_samples = torch.cat(far_samples)
    return far_samples

def generate_outlier(weight_yes, outlier_weight, threshold):
    boundary = ID_Cluster(weight_yes)
    distance = boundary @ outlier_weight.float().T
    idx = torch.argmax(distance, dim=1)
    random.seed(42)
    pseudo_label = []
    for b, i in zip(boundary, idx):
        f = random.uniform(0.0, 0.5)
        new_embe = b * f + outlier_weight[i] * (1-f)
        pseudo_label.append(new_embe.reshape(1, 512))
    pseudo_label = torch.cat(pseudo_label, dim=0)
    pseudo_label /= pseudo_label.norm(dim=-1, keepdim=True)
    dis = pseudo_label.float() @ weight_yes.T
    output = torch.max(dis, dim=1)[0]
    pseudo_label = pseudo_label[output < threshold]
    return pseudo_label


if __name__ == '__main__':
    args = parse_arguments()
    model, process_train, process_test = open_clip.create_model_and_transforms(args.model_type,
                                                                               pretrained=args.checkpoint_path,
                                                                               device=args.device, freeze=False)

    dataset = ImageNet(preprocess_test = process_test, batch_size = args.batch_size)

    file_yes = f"weight/ID_weight_yes.pkl"
    file_no = f"weight/ID_weight_no.pkl"

    if os.path.exists(file_yes) is not True:
        print('Preparing ID weight...')
        weight_yes, weight_no = merge_yes_no_feature(dataset, model, args.device)
        print(weight_yes.shape)
        with open(file_yes, 'wb') as f:
            pickle.dump(weight_yes, f)
        with open(file_no, 'wb') as f:
            pickle.dump(weight_no, f)
        print('Done')
    else:
        with open(file_yes, 'rb') as f:
            weight_yes = pickle.load(f)
        with open(file_no, 'rb') as f:
            weight_no = pickle.load(f)

    out_file = f"weight/outlier_weight.pkl"
    if os.path.exists(out_file) is not True:
        print('Preparing outlier weight...')
        # with open(f"weight/OOD_weight_yes.pkl", 'rb') as f:
        #     outlier_weight = pickle.load(f).T
        outlier_weight = encoder_outlier_prompt(model, args.device, args.label_path)



        outlier_weight = prototype_learning(outlier_weight, args.n_prototypes)

        outlier_weight, thres = refinement(weight_yes, outlier_weight, quantile=args.quantile)

        pesudo_weight = generate_outlier(weight_yes, outlier_weight, threshold=thres)

        outlier_weight = torch.cat([outlier_weight.cuda().float(), pesudo_weight.float()])

        with open(out_file, 'wb') as f:
            pickle.dump(outlier_weight, f)
        print('Done')
    else:
        print('Already got weights.')