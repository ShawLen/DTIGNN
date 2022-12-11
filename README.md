# DTIGNN
This is the PyTorch implementation of DTIGNN in the following paper: [Modeling Network-level Traffic Flow Transitions on Sparse Data](https://arxiv.org/abs/2208.06646).
```
@inproceedings{lei2022modeling,
  title={Modeling Network-level Traffic Flow Transitions on Sparse Data},
  author={Lei, Xiaoliang and Mei, Hao and Shi, Bin and Wei, Hua},
  booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={835--845},
  year={2022}
}
```
Usage and more information can be found below.

## Usage

* Step 1: Process datasets
```
# 1. Randomly mask several intersections.
# 2. Generate train, validation and test datasets.
python prepareData.py --config configurations/DTIGNN.conf
```

* Step 2: Train and test model
```
python train_DTIGNN.py --config configurations/DTIGNN.conf
```

## Datasets
Both synthetic and real-world data are included, which contains two networks at different scales: 16 intersections and 196 intersections. All the data can be found in ``data/uniform_4x4`` && ``data/hz_4x4`` && ``data/manhattan_28x7``.

The settings for each model can be found in the "configurations" folder.
