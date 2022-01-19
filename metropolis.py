from sklearn.mixture import GaussianMixture
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from scipy.stats import bernoulli

from ResNet.Models import ResNet
from AutoEncoder.Models import AutoEncoder
from GenerativeClassifier import GenerativeClassifier as GC

import torch
import torch.nn as nn
from torch import distributions as D
from torch.distributions import MultivariateNormal as MVN

import torchvision as tv
from torchvision.transforms import transforms

import pickle
from tqdm import tqdm
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_mnist(batch_size):
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

  train_loader = torch.utils.data.DataLoader(
    train_set, batch_size
  )
  test_loader = torch.utils.data.DataLoader(
    test_set, batch_size
  )

  return train_loader, test_loader
# train_loader, test_loader = get_mnist(1)

model = ResNet(1, 10).to(device)
model.load_state_dict(torch.load('chkpt/resnet_mnist.tar')['model_state_dict'])
model.eval()
with open('chkpt/encoded_mean_var_mnist_result.pickle', 'rb') as in_file:
  d = pickle.load(in_file)
  mean = d['mean'].to(device)
  var = d['var'].to(device)

enc = AutoEncoder(784, 20, 15).to(device)
enc.load_state_dict(torch.load('chkpt/encoder_mnist.tar')['model_state_dict'])
enc.eval();

gen_clf = GC(model, enc, 28*28, 10, mean, var)



def acceptance(x, y):
  a = gen_clf(gen_clf.enc.decode(x).view(1,1,28,28))
  b = gen_clf(gen_clf.enc.decode(y).view(1,1,28,28))
  p = a - b
  u = np.random.random()
  return p >= np.log(u)


def nextState(x):
  K = 20
  y = np.random.multivariate_normal(x.view(20).detach().cpu().numpy(), np.eye(K))
  y = torch.Tensor(y).to(device).view(1,1,20)
  accept = acceptance(y, x)
  if accept:
    return y
  else:
    return x

def metropolis(x, n_samples=1000):
  logprobs = np.zeros(n_samples)
  x_samples = np.zeros((n_samples, 784))
  with tqdm(total=n_samples) as progress:
    for i in range(n_samples):
      logprobs[i] = gen_clf(gen_clf.enc.decode(x).view(1,1,28,28))
      x = nextState(x)
      x_samples[i, :] = gen_clf.enc.decode(x).detach().cpu().view(28*28).numpy().copy()
      progress.update()
  return x_samples, logprobs

initial = mean[0]
initial += torch.randn(20).to(device)
# plt.imshow(initial.detach().cpu().numpy().reshape(28,28), cmap='gray')
# plt.show()
samples, logprobs = metropolis(initial.view(1,1,20).to(device), 5000)
# samples, logprobs = metropolis(mean[0].view(1,1,28,28).to(device), 100)
# samples, logprobs = metropolis(torch.randn(1,1,28,28).to(device), 100)


out_dict = {
  'samples': samples,
  'logprobs': logprobs
}

with open('bad_samples_logprobs.pickle','wb') as out:
  pickle.dump(out_dict, out)
