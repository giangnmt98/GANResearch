"""
Module for creating various types of GAN models based on configuration settings.
"""

from ganresearch.models.cgan import CGAN
from ganresearch.models.dcgan import DCGAN
from ganresearch.models.wgan_gp import WGANGP
from ganresearch.utils.utils import create_logger

# Initialize logger for logging messages and events
logger = create_logger()


class GANFactory:
    def __init__(self, config):
        """
        Initialize the GANFactory with a given configuration.

        Parameters:
            config (dict): Configuration dictionary specifying model parameters.
        """
        self.config = config

    def create_model_gan(self):
        """
        Create and return a GAN model based on the specified type in the config.

        The function supports creating DCGAN, WGAN-GP, and CGAN models.

        Returns:
            BaseGAN: An instantiated GAN model based on the specified type.

        Raises:
            ValueError: If the specified model type in the config is not supported.
        """
        # Extract the model name from the configuration dictionary
        model_name = self.config["model"]["name"]

        # Check the model name and create the corresponding model instance
        if model_name == "dcgan":
            return DCGAN(self.config)  # Create and return a DCGAN model
        elif model_name == "wgan_gp":
            return WGANGP(self.config)  # Create and return a WGAN-GP model
        elif model_name == "cgan":
            return CGAN(self.config)  # Create and return a CGAN model
        else:
            # Raise an error if the model name is not recognized
            raise ValueError(f"Unsupported GAN type: {model_name}")
