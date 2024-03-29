from __future__ import print_function

import argparse
import os

import numpy as np
import torch
import torchvision.datasets as datasets

from MnistVAE import MnistVAE
from torch import optim
from torchvision import transforms
from tqdm import tqdm
from utils import *

def main():
    #train args
    parser = argparse.ArgumentParser(description="Variational Autoencoder")
    parser.add_argument("--datadir", default='./', help="path to dataset")
    parser.add_argument("--outdir", default="./result", help="directory to output images")
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
        )
    parser.add_argument("--epochs", type=int, default=2, metavar="N", help="number of epoch to train (default: 50)")
    parser.add_argument("--lr", type=float, default=0.0005, metavar="LR", help="learning rate (default: 0.0005)")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        metavar="N",
        help="number of dataloader workers if device is CPU (default: 16)",
        )
    parser.add_argument("--seed", type=int, default=16, metavar="S", help="random seed (default: 16)")
    parser.add_argument("--latent-size", type=int, default=32, help="Latent size")
    parser.add_argument("--hsize", type=int, default=100, help="h size")
    parser.add_argument("--dataset", type=str, default="MNIST", help="(MNIST||FMNIST)")
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')

    args = parser.parse_args()

    torch.random.manual_seed(args.seed)
    latent_size = args.latent_size
    dataset = args.dataset
    assert dataset in ["MNIST"]
    if not (os.path.isdir(args.datadir)):
        os.makedirs(args.datadir)
    if not (os.path.isdir(args.outdir)):
        os.makedirs(args.outdir)
    if not (os.path.isdir(args.outdir)):
        os.makedirs(args.outdir)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(
        "batch size {}\nepochs {}\nAdam lr {}\nusing device {}\n".format(
            args.batch_size, args.epochs, args.lr, device.type
        )
    )
    #Build train and test set data loaders
    if dataset == "MNIST":
        image_size = 28
        num_chanel = 1
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                args.datadir, train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                args.datadir, train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])
            ),
            batch_size=10000,
            shuffle=False,
            num_workers=args.num_workers,
        )
        test_loader2 = torch.utils.data.DataLoader(
            datasets.MNIST(
                args.datadir, train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])
            ),
            batch_size=64,
            shuffle=False,
            num_workers=args.num_workers,
        )
        model = MnistVAE(image_size=28, latent_size=args.latent_size, hidden_size = 200, device=device).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
        ite=0
        for epoch in range(args.epochs):
            total_loss = 0.0
            print('Training ...')
            for batch_idx, (data, y) in tqdm(enumerate(train_loader, start=0)):
                data = data.to(device)
                # vae reconstruction
                image_batch_recon, latent_mu, latent_logvar = model(data)
                # vae loss computation
                loss = vae_loss(image_batch_recon, data, latent_mu, latent_logvar)
                #backpropagation
                optimizer.zero_grad()
                loss.backward()
                #one step of the optimizer
                optimizer.step()
                total_loss += loss.item()
                if batch_idx % args.log_interval == 0:
                  print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                      epoch, batch_idx * len(data), len(train_loader.dataset),
                      100. * batch_idx / len(train_loader),
                      loss.item() / len(data)))
  
        print("Epoch:" + str(epoch+1) + "Loss: " + str(total_loss/len(train_loader.dataset)))

if __name__ == "__main__":
    main()


