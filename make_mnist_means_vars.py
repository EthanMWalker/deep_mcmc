import torch
import torch.nn as nn
from torch import distributions
import torchvision as tv
from torchvision.transforms import transforms

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

from SimCLR.Models import ResNet
from train_resnet_cifar import train

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_mnist():
  transform = transforms.Compose(
    [
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.30811,))
    ]
  )

  train_set = tv.datasets.MNIST(
    '../data/', train=True, download=True, transform=transform
  )
  test_set = tv.datasets.MNIST(
    '../data/', train=False, download=True, transform=transform
  )

  return train_set, test_set


if __name__ == '__main__':
  train_set, test_set = get_mnist()
  xs = torch.empty(len(train_set), 28*28)
  ys = torch.empty(len(train_set))

  mean = torch.zeros(10,28*28).to(device)
  var = torch.zeros(10,28*28, 28*28).to(device)

  for i, item in enumerate(train_set):
    xs[i] = item[0].view(28*28)
    ys[i] = item[1]


  for i in range(10):
    mask = ys == i
    mean[i] = xs[mask].mean(axis=0)
    var[i] = xs[mask].var(axis=0)
  
  out_dict = {
    'mean': mean.detach().cpu().numpy(),
    'var': var.detach().cpu().numpy()
  }

  with open(f'resnet_chkpt/mean_var_cifar_result.pickle','wb') as out:
    pickle.dump(out_dict, out)