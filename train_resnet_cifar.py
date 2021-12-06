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

def train(model, train_loader, lr, num_epochs=10, save_iters=5):

  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  criterion = nn.CrossEntropyLoss()
  losses = []
  running_loss = 0
  i = 0

  with tqdm(total=len(train_loader)*num_epochs) as progress:
    for epoch in range(num_epochs):
      for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out,y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        progress.set_description(f'epoch: {epoch} loss: {loss.item():.2f}')
        progress.update()

        if i % 100 == 99:
          losses.append(running_loss / 100)
          running_loss = 0
        i += 1

      if epoch % save_iters == 0:
        torch.save(
          {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss
          }, f'resnet_chkpt/resnet_cifar_{epoch}.tar'
        )
  
  return model, losses

def test(model, test_loader):
  actual = []
  predicted = []

  correct = 0
  total = 0
  with torch.no_grad():
    for data in test_loader:
      X_test, y_test = data[0].to(device), data[1].to(device)

      outputs = model(X_test)
      _, predict = torch.max(outputs.data, 1)
      total += y_test.shape[0]
      correct += (predict == y_test).sum().item()

      actual.extend(y_test.cpu())
      predicted.extend(predict.cpu())

  return predicted, actual


if __name__ == '__main__':
  train_loader, test_loader = get_cifar(512)
  n_epochs = 200
  lr = 1e-5

  model = ResNet(3, 10).to(device)
  

  model, losses = train(model, train_loader, lr, n_epochs, 10)

  torch.save(
    {
    'model_state_dict': model.state_dict()
    }, f'resnet_chkpt/resnet_cifar.tar'
  )
  # model.load_state_dict(torch.load('chkpt/test.tar')['model_state_dict'])


  actual, predicted = test(model, test_loader)

  classes = (
    'plane', 'car', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
  )
  matrix = confusion_matrix(actual, predicted, labels=[0,1,2,3,4,5,6,7,8,9])

  figure = plt.figure(figsize=(16,9))
  ax = figure.add_subplot(111)
  disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=classes)
  disp.plot(ax=ax)
  plt.savefig('resnet_chkpt/resnet_cifar_conf')
  plt.clf()

  plt.plot(np.linspace(0,n_epochs,len(losses)),losses, label='mean loss')
  plt.xlabel('epochs')
  plt.ylabel('loss')
  plt.legend()
  plt.savefig('resnet_chkpt/resnet_cifar_loss')

  out_dict = {
    'losses': losses,
    'predicted': predicted,
    'actual': actual
  }

  with open(f'resnet_chkpt/resnet_cifar_result.pickle','wb') as out:
    pickle.dump(out_dict, out)