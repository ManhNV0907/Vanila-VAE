import torch 
import torch.nn as nn
from utils import *

class MnistVAE(nn.Module):
    def __init__(self, image_size, hidden_size, latent_size, device):
      super(MnistVAE, self).__init__()
      self.image_size = image_size**2
      self.hidden_size = hidden_size
      self.latent_size = latent_size
      self.device = device
      # self.encoder = nn.Sequential(
      #     nn.Linear(self.image_size, hidden_size),
      #     nn.LeakyReLu(0.2,True)
      #     nn.Linear(hidden_size, latent_size),
      # )
      self.encoder1 = nn.Sequential(
        nn.Linear(self.image_size, hidden_size),
        nn.ReLU(True),
        nn.Linear(hidden_size, latent_size),
      )
      self.encoder2 = nn.Sequential(
        nn.Linear(self.image_size, hidden_size),
        nn.ReLU(True),
        nn.Linear(hidden_size, latent_size),
      )
    # def encoder(x):
    #   self.fc1 = nn.Linear(self.image_size, hidden_size)
    #   self.fc2 = nn.Linear(hidden_size, latent_size)
    #   self.fc3 = nn.Linear(hidden_size, latent_size)
    #   x = F.relu(self.fc1(x.view(-1, 28 * 28)))
    #   mu = self.fc2(x)
    #   log_sigma = self.fc3(x)
    #   return mu, log_sigma
      
      self.decoder = nn.Sequential(
        nn.Linear(latent_size, hidden_size),
        nn.ReLU(True),
        nn.Linear(hidden_size, 2 * hidden_size),
        nn.ReLU(True),
        nn.Linear(2 * hidden_size, 4 * hidden_size),
        nn.ReLU(True),
        nn.Linear(4 * hidden_size, self.image_size),
        nn.Sigmoid(),
      )

    def forward(self, x):
        latent_mu = self.encoder1(x.view(-1, 28 * 28))
        latent_logvar =self.encoder2(x.view(-1, 28 * 28))
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar
    
    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    


