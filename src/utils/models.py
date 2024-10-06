import os
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F

def torch_save(classifer, save_path="./"):
    if os.path.dirname(save_path) != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(classifer.cpu(), f)


def torch_load(save_path, device=None):
    with open(save_path, 'rb') as f:
        classifier = pickle.load(f)
    if device is not None:
        classifier = classifier.to(device)
    return classifier

class ViT_Classifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head_yes, classification_head_no, scale):
        super().__init__()
        self.image_encoder = image_encoder
        flag = True
        self.fc_yes = nn.Parameter(classification_head_yes, requires_grad=flag)
        self.fc_no = nn.Parameter(classification_head_no, requires_grad=flag)
        self.scale = scale

    def set_frozen(self, module):
        for module_name in module.named_parameters():
            module_name[1].requires_grad = False

    def set_learnable(self, module):
        for module_name in module.named_parameters():
            module_name[1].requires_grad = True

    def forward(self, x):
        inputs = self.image_encoder(x)
        inputs_norm = F.normalize(inputs, dim=-1)
        fc_yes = F.normalize(self.fc_yes, dim=-1)
        fc_no = F.normalize(self.fc_no, dim=-1)

        logits_yes = self.scale * inputs_norm @ fc_yes.T
        logits_no = self.scale * inputs_norm @ fc_no.T
        return logits_yes, logits_no

    def save(self, path="./"):
        torch_save(self, path)

    @classmethod
    def load(cls, filename="./", device=None):
        return torch_load(filename, device)
