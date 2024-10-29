"""
This module defines an abstract base class for Generative Adversarial Networks (GANs).
"""

from abc import ABC, abstractmethod


class BaseGAN(ABC):
    """
    Base class for Generative Adversarial Networks (GANs) that serves as a template
    for building and training various types of GANs.
    """

    def __init__(self, config):
        """
        Initializes the GAN with configurations, builds generator and discriminator,
        and sets up optimizers.

        Args:
            config (dict): Configuration dictionary containing the parameters for
                           the device, noise dimension, and others.
        """
        self.config = config
        self.device = config["training"]["device"]
        self.noise_dim = config["training"]["noise_dimension"]

        # Initialize generator and discriminator
        self.generator = self.build_generator(
            nz=self.config["training"]["noise_dimension"],
            ngf=self.config["dataset"]["image_size"],
            nc=self.config["dataset"]["channels"],
        ).to(self.device)

        self.discriminator = self.build_discriminator(
            ndf=self.config["training"]["noise_dimension"],
            nc=self.config["dataset"]["channels"],
        ).to(self.device)

        # Create optimizers for generator and discriminator
        self.gen_optimizer = None
        self.disc_optimizer = None

    @abstractmethod
    def build_generator(self, nz, ngf, nc):
        """
        Builds the generator network.
        """
        pass

    @abstractmethod
    def build_discriminator(self, ndf, nc):
        """
        Builds the discriminator network.
        """
        pass

    @abstractmethod
    def loss(self, real_output, fake_output):
        """
        Computes the loss for the discriminator and generator.

        Args:
            real_output (torch.Tensor): Discriminator output for real images.
            fake_output (torch.Tensor): Discriminator output for fake images.
        """
        pass
