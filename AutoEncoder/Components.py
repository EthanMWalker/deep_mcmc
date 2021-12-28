import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):

  def __init__(self, in_dim, out_dim, layers):
    super(Encoder, self).__init__()

    steps = np.linspace(in_dim,out_dim, layers,endpoint=True).astype(int)

    linears = [ nn.Linear(a,b) for a,b in zip(steps[:-1],steps[1:])]
    relus = [nn.ReLU() for _ in range(len(steps))]

    inside = [val for pair in zip(linears, relus) for val in pair]

    self.encoder = nn.Sequential(
      *inside[:-1]
    )
  
  def forward(self,x):
    return self.encoder(x)

class Decoder(nn.Module):

  def __init__(self, in_dim, out_dim, layers):
    super(Decoder, self).__init__()

    steps = np.linspace(in_dim,out_dim, layers,endpoint=True).astype(int)

    linears = [ nn.Linear(a,b) for a,b in zip(steps[:-1],steps[1:])]
    relus = [nn.ReLU() for _ in range(len(steps-1))]
    relus += [nn.Sigmoid()]

    inside = [val for pair in zip(linears, relus) for val in pair]

    self.decoder = nn.Sequential(
      *inside
    )
  
  def forward(self,x):
    return self.decoder(x)
    


