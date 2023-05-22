# RL-Algorithms
PyTorch implementations of a variety of reinforcement learning algorithms

## Requirements
- [Pytorch](https://pytorch.org)
- pip install numpy
- pip install gymnasium
- pip instsall wandb

## How to run
```
$ python train.py config/train_vpg_cartpole.py
```
New config files can be created or overwritten in the cmd:
```
$ python train.py config/train_vpg_cartpole.py --batch_size=64
```

To test a model, run:
```
$ python test.py path/to/checkpoint.pt
```
