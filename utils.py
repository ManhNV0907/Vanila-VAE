import numpy as np 
import torch 
import torch.nn.functional as F

def save_model(model, optimier, epoch, folder):
    dictionary = {}
    dictionary["epoch"] = epoch
    dictionary["model"] = model.state_dict()
    dictionary["optimizer"] = optimizer.state_dict()
    torch.save(dictionary, folder + "/model.pth")

def load_model(folder):
    dictionary = torch.load(folder + "model.pth")
    return (
        dictionary["epoch"],
        dictionary["model"],
        dictionary["optimizer"],
    )

def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD