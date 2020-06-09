# Self-Distillation

This is the repository of Self-Distillation codes in the journal version.

## Requirements

Install the required packages. 

```
pip install torch torchvision
```

## Experiments on CIFAR100

Train a ResNet152 model with self-distillation on CIFAR100 dataset.

```python
python train.py --model=resnet152
```
