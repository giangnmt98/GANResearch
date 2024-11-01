"""
This module defines an abstract base class for Generative Adversarial Networks (GANs).
"""

from abc import ABC


class BaseGAN(ABC):
    """
    Base class for Generative Adversarial Networks (GANs) that serves as a template
    for building and training various types of GANs.
    """

    def __init__(self, config, losses):
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
        self.generator = None
        self.discriminator = None

        # Create optimizers for generator and discriminator
        self.gen_optimizer = None
        self.disc_optimizer = None
        self.losses = losses
