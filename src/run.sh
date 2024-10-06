python prepare_weight.py --checkpoint_path=./clipn_checkpoint.pt --label_path=./outlier_label/nonus.npy
python inference.py --checkpoint_path=./clipn_checkpoint.pt --dataset_path=../Datasets
    
