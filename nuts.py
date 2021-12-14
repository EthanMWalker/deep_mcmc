import torch
from ResNet.Models import ResNet
from GenerativeClassifier import GenerativeClassifier

import pyro
from pyro.distributions import Normal
from pyto.infer import NUTS
from pyro.infer.mcmc.util import initialize_model, summary

import pickle


def my_model():
  pass


