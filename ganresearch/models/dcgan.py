"""
This module defines the DCGAN, Generator, and Discriminator classes used for
implementing Deep Convolutional Generative Adversarial Networks.
"""

from abc import ABC

import torch
import torch.nn as nn

from ganresearch.models.base_gan import BaseGAN


class DCGAN(BaseGAN, ABC):
    """
    A DCGAN class that extends the BaseGAN and abstract base class.
    """

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
        self.discriminator.apply(self.weights_init)
        # Initialize the generator model and move it to the specified device
        self.generator = Generator(self.config).to(self.device)
        self.generator.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.zeros_(m.bias)


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

        # Define the sequence of layers for the generator
        self.main = nn.Sequential(
            # Input is Z, going into a convolution
            nn.ConvTranspose2d(self.noise_dimension, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # State size: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # State size: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # State size: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # State size: (ngf) x 32 x 32
            nn.ConvTranspose2d(self.ngf, self.image_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
            # Output state size: 3 x 64 x 64
        )

    def forward(self, input):
        """
        Perform a forward pass of the generator.

        Args:
            input (torch.Tensor): Input tensor for the generator.

        Returns:
            torch.Tensor: Output tensor from the generator.
        """
        # Check if CUDA is available and multiple GPUs are specified
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

        # Define the sequence of layers for the discriminator
        self.main = nn.Sequential(
            # Input state size: 3 x 64 x 64
            nn.Conv2d(self.image_channels, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        """
        Perform a forward pass of the discriminator.

        Args:
            input (torch.Tensor): Input tensor for the discriminator.

        Returns:
            torch.Tensor: Output tensor from the discriminator.
        """
        # Check if CUDA is available and multiple GPUs are specified
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        # Flatten the output before returning
        return output.view(-1, 1).squeeze(1)
