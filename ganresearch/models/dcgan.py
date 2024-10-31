from abc import ABC

import torch
import torch.nn as nn

from ganresearch.models.base_gan import BaseGAN


class DCGAN(BaseGAN, ABC):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.config = config
        self.device = device
        self.discriminator = Discriminator(self.config).to(self.device)
        self.generator = Generator(self.config).to(self.device)



    def loss(self, real_output, fake_output):
        criterion = nn.BCELoss()
        real_loss = criterion(real_output, torch.ones_like(real_output))
        fake_loss = criterion(fake_output, torch.zeros_like(fake_output))
        return real_loss + fake_loss


class Generator(nn.Module):
    def __init__(self, ngpu, config):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.config = config
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.config.nz, self.config.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.config.ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.config.ngf * 8, self.config.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.config.ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.config.ngf * 4, self.config.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.config.ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.config.ngf * 2, self.config.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.config.ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.config.ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, ngpu, config):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.config = config
        self.main = nn.Sequential(
            nn.Conv2d(3, self.config.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.config.ndf, self.config.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.config.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.config.ndf * 2, self.config.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.config.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.config.ndf * 4, self.config.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.config.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.config.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)