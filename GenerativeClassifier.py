import torch
import torch.nn as nn
from torch.distributions import Normal as N
from torch.distributions import MultivariateNormal as MVN

class GenerativeClassifier(nn.Module):

  def __init__(self, model, encoder, n_dim, n_classes, means, vars):
    super(GenerativeClassifier, self).__init__()
    self.clf = model
    self.enc = encoder
    self.dims = n_dim
    self.n_classes = n_classes
    self.means = means
    self.vars = vars
  
  def pdf(self, x):
    pis = torch.log(self.clf(x))
    x = torch.flatten(x,1)
    probs = torch.zeros(x.size(0)).to(x.device)
    for i in range(self.n_classes):
      probs += pis[:,i]
      probs += N(self.means[i], self.vars[i]).log_prob(self.enc.encode(x)).mean(axis=1)
    
    return probs

  def forward(self, x):
    pis = torch.log(self.clf(x))
    x = torch.flatten(x,1)
    probs = torch.zeros(x.size(0)).to(x.device)
    for i in range(self.n_classes):
      probs += pis[:,i]
      probs += MVN(
        self.means[i], torch.diag(self.vars[i])
      ).log_prob(self.enc.encode(x))
    
    return probs
