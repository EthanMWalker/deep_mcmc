import torch
import torchvision as tv
from torchvision.transforms import transforms
import pickle

from AutoEncoder.Models import AutoEncoder

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

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
  latent_size = 10

  train_set, test_set = get_mnist()
  xs = torch.empty(len(train_set), latent_size)
  ys = torch.empty(len(train_set))

  enc = AutoEncoder(784, latent_size, 15).to(device)
  enc.load_state_dict(torch.load('chkpt/encoder_mnist.tar')['model_state_dict'])
  enc.eval();

  mean = torch.zeros(10,latent_size).to(device)
  var = torch.ones(10,latent_size).to(device)
  # var = torch.zeros(10,28*28, 28*28).to(device)

  for i, item in enumerate(train_set):
    xs[i] = enc.encode(item[0].view(28*28).to(device))
    ys[i] = item[1]


  for i in range(10):
    mask = ys == i
    mean[i] = xs[mask].mean(axis=0)
    var[i] = xs[mask].var(axis=0) + 1e-20
  
  out_dict = {
    'mean': mean.detach().cpu(),
    'var': var.detach().cpu()
  }

  with open(f'chkpt/encoded_mean_var_mnist_result.pickle','wb') as out:
    pickle.dump(out_dict, out)