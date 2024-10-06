# Zero-Shot Out-Of-Distribution Detection with Outlier Label Exposure
By Choubo Ding, Guansong Pang

Official PyTorch implementation of the paper “Zero-Shot Out-Of-Distribution Detection with Outlier Label Exposure”

Code is modified from [CLIPN](https://github.com/xmed-lab/CLIPN).

## Installation
   ```bash
   git clone https://github.com/Choubo/OLE.git
   cd OLE
   conda create -n ole python=3.8
   conda activate ole
   pip install -r requirements.txt
   ```

## Dataset Preparation

Our experiments utilize the following datasets:

#### In-distribution
- **ImageNet-1K**: The ILSVRC-2012 version, serving as our primary in-distribution dataset. It can be downloaded [here](https://image-net.org/challenges/LSVRC/2012/index.php#)

#### Out-of-distribution
We use subsampled versions of the following datasets, with classes overlapping ImageNet-1K removed:

- iNaturalist
- SUN
- Places
- Texture

For detailed download instructions and preprocessing steps, please refer to the [MOS](https://github.com/deeplearning-wisc/large_scale_ood#out-of-distribution-dataset).

## Quick Start
1. Download and place the checkpoint:
   - Download the checkpoint from [this link](https://drive.google.com/file/d/12iF9SppxRrNR7cBhlFzvJQvkkaXQCz7Z/view?usp=sharing).
   - Place the downloaded checkpoint in the `src` folder:
     ```bash
     mv path/to/downloaded/clipn_checkpoint.pth src/
     ```

2. Run outlier prototype learning and OLE evaluation:
   ```bash
   cd src
   bash run.sh
   ```

This script will execute the outlier prototype learning process and perform OLE evaluation.

## Citation

If you use our codebase, please cite our work:
```
@inproceedings{
ding2024zero,
title={Zero-Shot Out-of-Distribution Detection with Outlier Label Exposure},
author={Choubo Ding and Guansong Pang},
booktitle={2024 International Joint Conference on Neural Networks},
year={2024}
}
```