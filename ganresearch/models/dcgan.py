from abc import ABC

import torch
import torch.nn as nn

from ganresearch.models.base_gan import BaseGAN


class DCGAN(BaseGAN, ABC):
    def __init__(self, config):
        super().__init__(config)
        self.criterion = nn.BCEWithLogitsLoss()

    def build_generator(self, nz, ngf, nc):
        """
        Creates a generator network using PyTorch Sequential API.

        Args:
            nz (int): Size of the latent vector (noise input).
            ngf (int): Size of feature maps in the generator.
            nc (int): Number of channels in the output image (1 for grayscale, 3 for RGB).

        Returns:
            nn.Sequential: Generator network.
        """
        return nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def build_discriminator(self, ndf, nc):
        """
        Creates a discriminator network using PyTorch Sequential API.

        Args:
            ndf (int): Size of feature maps in the discriminator.
            nc (int): Number of channels in the input image (1 for grayscale, 3 for RGB).

        Returns:
            nn.Sequential: Discriminator network.
        """
        return nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),  # (ndf) x 32 x 32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),  # (ndf*2) x 16 x 16
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),  # (ndf*4) x 8 x 8
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),  # (ndf*8) x 4 x 4
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),  # 1 x 1 x 1
            nn.Sigmoid(),  # Output a scalar value between 0 and 1
        )

    def loss(self, real_output, fake_output):
        real_loss = self.criterion(real_output, torch.ones_like(real_output))
        fake_loss = self.criterion(fake_output, torch.zeros_like(fake_output))
        return real_loss + fake_loss
