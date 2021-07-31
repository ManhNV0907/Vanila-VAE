import torch 
import torch.nn as nn
from utils import *

class Encoder(nn.Module):
    def __init__(self, image_size, hidden_size, latent_size):
        super().__init__()
        self.image_size = image_size ** 2
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.main = nn.Sequential(
            nn.Linear(self.image_size, 4 * self.hidden_size),
            nn.LeakyReLU(0.2, True),
            nn.Linear(4 * self.hidden_size, 2 * self.hidden_size),
            nn.LeakyReLU(0.2, True),
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.LeakyReLU(0.2, True),
        )
        self.fc_mu = nn.Linear(self.hidden_size, self.latent_size)
        self.fc_logvar = nn.Linear(self.hidden_size, self.latent_size)
    def foward(self, input):
        output = self.main(input.view(-1, self.image_size))
        x_mu = self.fc_mu(output)
        x_logvar = self.fc_logvar(output)
        return x_mu, x_logvar
    
class Decoder(nn.Module):
    def __init__(self, image_size, hidden_size, latent_size):
        super(Decoder, self).__init__()
        self.image_size = image_size ** 2
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.main = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.ReLU(True),
            nn.Linear(2 * hidden_size, 4 * hidden_size),
            nn.ReLU(True),
            nn.Linear(4 * hidden_size, self.image_size),
            nn.ReLU(True),
        )

    def forward(self, input):
        return self.main(input)

class MnistVAE(nn.Module):
    def __init__(self, image_size, hidden_size, latent_size, device):
        super().__init__()
        self.image_size = image_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.device = device
        self.encoder = Encoder(image_size, hidden_size, latent_size)
        self.decoder = Decoder(image_size, hidden_size, latent_size)
    

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
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

    


