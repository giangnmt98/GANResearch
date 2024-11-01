from abc import ABC

import torch.nn as nn

from ganresearch.models.base_gan import BaseGAN


class WGANGP(BaseGAN, ABC):

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
        self.discriminator = Critic(self.config).to(self.device)

        # Initialize the generator model and move it to the specified device
        self.generator = Generator(self.config).to(self.device)


class Generator(nn.Module):
    """
    Generator model for the DCGAN.

    Attributes:
        config (dict): Configuration settings.
        ngpu (int): Number of GPUs to use.
        noise_dimension (int): Dimension of the noise input.
        self.ngf (int): Number of generator filters.
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
            nn.ConvTranspose2d(self.noise_dimension, self.ngf * 8, 4, 1, 0, bias=False),
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


class Critic(nn.Module):
    """
    Discriminator model for the DCGAN.

    Attributes:
        config (dict): Configuration settings.
        ngpu (int): Number of GPUs to use.
        self.ndf (int): Number of discriminator filters.
        main (nn.Sequential): Sequence of layers.
    """

    def __init__(self, config):
        """
        Initialize the Discriminator model with configuration settings.

        Args:
            config (dict): Configuration settings.
        """
        super(Critic, self).__init__()

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
            nn.Conv2d(self.image_channels, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 4, 1, 4, 1, 0, bias=False),
        )

    def forward(self, input):
        return self.main(input).view(-1)
