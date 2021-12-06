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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_cifar(batch_size):
  transform = transforms.Compose(
    [
      transforms.ToTensor(),
      transforms.Normalize((.5,.5,.5),(.5,.5,.5))
    ]
  )

  train_set = tv.datasets.CIFAR10(
    '../data/', train=True, download=True, transform=transform
  )
  test_set = tv.datasets.CIFAR10(
    '../data/', train=False, download=True, transform=transform
  )

  train_loader = torch.utils.data.DataLoader(
    train_set, batch_size
  )
  test_loader = torch.utils.data.DataLoader(
    test_set, batch_size
  )

  return train_loader, test_loader

def train(train_loader):

  means = torch.zeros(10,28,28).to(device)
  vars = torch.zeros(10,28,28).to(device)

  with tqdm(total=len(train_loader)) as progress:
    for x, y in train_loader:
      x = x.to(device)
      y = y.to(device)

      for i in range(10):
        mask = y == i
        print(x[mask])
        print(x[mask].size())
      break

      progress.update()

  return means, vars

if __name__ == '__main__':
  train_loader, test_loader = get_cifar(2)

  means, vars = train(train_loader)

  out_dict = {
    'means': means,
    'vars': vars
  }

  with open(f'resnet_chkpt/resnet_cifar_result.pickle','wb') as out:
    pickle.dump(out_dict, out)