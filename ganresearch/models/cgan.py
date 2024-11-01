from abc import ABC

import torch
import torch.nn as nn

from ganresearch.models.base_gan import BaseGAN


class CGAN(BaseGAN, ABC):

    def __init__(self, config, losses):
        """
        Initialize the DCGAN model with configuration and device settings.

        Args:
            config (dict): Configuration settings.
            device (str): The device (e.g., "cpu" or "cuda") to run the model on.
        """
        super().__init__(config, losses)

        # Store the configuration settings and device
        self.config = config
        self.device = self.config["training"]["device"]

        # Initialize the discriminator model and move it to the specified device
        self.discriminator = Discriminator(self.config).to(self.device)

        # Initialize the generator model and move it to the specified device
        self.generator = Generator(self.config).to(self.device)


class Generator(nn.Module):
    """
    Generator model for the DCGAN.

    Attributes:
        config (dict): Configuration settings.
        ngpu (int): Number of GPUs to use.
        noise_dimension (int): Dimension of the noise input.
        ngf (int): Number of generator filters.
        main (nn.Sequential): Sequence of layers.
    """

    def __init__(self, config):
        """
        Initialize the Generator model with configuration settings.

        Args:
            config (dict): Configuration settings.
        """
        super(Generator, self).__init__()

        # Store the configuration settings
        self.config = config

        # Number of GPUs to use
        self.ngpu = self.config["training"]["number_gpu"]

        # Noise dimension input
        self.noise_dimension = self.config["training"]["noise_dimension"]

        # Number of generator feature maps
        self.ngf = int(self.config["training"]["ngf"])

        # Number of image channels
        self.image_channels = int(self.config["dataset"]["channels"])
        # Number of classes
        self.num_classes = int(self.config["dataset"]["num_classes"])
        # Label embedding
        self.label_emb = nn.Embedding(self.num_classes, self.num_classes)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(
                self.noise_dimension + self.num_classes,
                self.ngf * 8,
                4,
                1,
                0,
                bias=False,
            ),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf, self.image_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        label_embedding = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        input = torch.cat((noise, label_embedding), 1)
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Discriminator(nn.Module):
    """
    Discriminator model for the DCGAN.

    Attributes:
        config (dict): Configuration settings.
        ngpu (int): Number of GPUs to use.
        ndf (int): Number of discriminator filters.
        main (nn.Sequential): Sequence of layers.
    """

    def __init__(self, config):
        """
        Initialize the Discriminator model with configuration settings.

        Args:
            config (dict): Configuration settings.
        """
        super(Discriminator, self).__init__()

        # Store configuration settings
        self.config = config

        # Number of GPUs to use
        self.ngpu = self.config["training"]["number_gpu"]

        # Number of discriminator feature maps
        self.ndf = int(self.config["training"]["ndf"])
        # Number of image channels
        self.image_channels = int(self.config["dataset"]["channels"])
        # Number of classes
        self.num_classes = int(self.config["dataset"]["num_classes"])

        self.label_emb = nn.Embedding(self.num_classes, self.num_classes)
        self.main = nn.Sequential(
            nn.Conv2d(
                self.image_channels + self.num_classes, self.ndf, 4, 2, 1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        label_embedding = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        label_embedding = label_embedding.expand(-1, -1, img.size(2), img.size(3))
        input = torch.cat((img, label_embedding), 1)
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output
