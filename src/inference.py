import os
import pickle
from tqdm import tqdm
import torch
import open_clip

from utils.cfg import parse_arguments
from utils.models import ViT_Classifier
from utils.datasets import Places, Textures, ImageNet, iNaturalist, SUN
from util import preprocess, cal_auc_fpr

def OLE_scoring(logits, logits_no, id_len):
    yesno = torch.cat([logits[:, :id_len].unsqueeze(-1), logits_no[:, :id_len].unsqueeze(-1)], -1)
    yesno = torch.softmax(yesno, dim=-1)[:, :id_len, 0]
    return list((yesno * torch.softmax(logits, -1)[:, :id_len]).sum(1).detach().cpu().numpy())

def run_dataset(dataset, model, id_len):
    OLE = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataset)):
            batch = preprocess(batch)
            inputs = batch["images"].cuda()
            logits, logits_no = model(inputs)
            OLE += OLE_scoring(logits, logits_no, id_len)
    return OLE

def infer(id_dataset, model, id_len, ood_dataset=None):
    model.eval()
    res = []
    # log_path = './log/' + args.exp_name
    with torch.no_grad():
        id_OLE = run_dataset(id_dataset, model, id_len)

        for name, ood_data in ood_dataset.items():
            ood_OLE = run_dataset(ood_data, model, id_len)

            auc, fpr = cal_auc_fpr(id_OLE, ood_OLE)
            res.append(["OLE", name, auc, fpr])

    ood_lis_epoch = res
    for lis in ood_lis_epoch:
        print(lis)

if __name__ == '__main__':
    args = parse_arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, process_train, process_test = open_clip.create_model_and_transforms(args.model_type,
                                                                               pretrained=args.checkpoint_path,
                                                                               device=args.device, freeze=False)

    dataset = ImageNet(preprocess_test=process_test, batch_size=args.batch_size, data_root=os.path.join(args.dataset_path, 'ImageNet-1k/val')).test_loader
    test_dataset = {
        "iNaturalist": iNaturalist(preprocess_test=process_test, batch_size=args.batch_size, data_root=os.path.join(args.dataset_path, 'iNaturalist')).test_loader,
        "SUN": SUN(preprocess_test=process_test, batch_size=args.batch_size, data_root=os.path.join(args.dataset_path, 'SUN')).test_loader,
        "Textures": Textures(preprocess_test=process_test, batch_size=args.batch_size, data_root=os.path.join(args.dataset_path, 'Textures')).test_loader,
        "Places": Places(preprocess_test=process_test, batch_size=args.batch_size, data_root=os.path.join(args.dataset_path, 'Places')).test_loader,
    }

    file_yes = f"weight/ID_weight_yes.pkl"
    file_no = f"weight/ID_weight_no.pkl"
    with open(file_yes, 'rb') as f:
        weight_yes = pickle.load(f)
    with open(file_no, 'rb') as f:
        weight_no = pickle.load(f)

    file_out = f"weight/outlier_weight.pkl"
    with open(file_out, 'rb') as f:
        outlier_weight = pickle.load(f)
    id_len = weight_yes.shape[0]
    weight_yes = torch.cat([weight_yes, outlier_weight])

    vit_classifier = ViT_Classifier(model.visual, weight_yes, weight_no, args.scale)


    model = vit_classifier.cuda()
    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)
    model = torch.nn.DataParallel(model, device_ids=devices)

    infer(dataset, model, id_len, test_dataset)

