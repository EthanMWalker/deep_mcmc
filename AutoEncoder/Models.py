from AutoEncoder.Components import *

class AutoEncoder(nn.Module):

  def __init__(self, in_dim, latent_dim, layers):
    super(AutoEncoder, self).__init__()

    self.encoder = Encoder(in_dim, latent_dim, layers)
    self.decoder = Decoder(latent_dim, in_dim, layers)
  
  def forward(self,x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
  
  def encode(self,x):
    return self.encoder(x)
  
  def decode(self,x):
    return self.decoder(x)