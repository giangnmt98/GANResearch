"""
This module defines classes and functions to compute and handle losses for
various GAN models, including support for LeCam regularization and Exponential
Moving Averages (EMA) of discriminator and generator losses.
"""

import torch
import torch.nn.functional as F


class Losses:
    """
    Class for managing and computing loss functions used in GAN training.

    Attributes:
        config (dict): Configuration parameters for the loss function.
        use_lecam (bool): Flag indicating whether to use LeCam regularization.
        lecam_ratio (float): Ratio for LeCam regularization if applicable.
        ema (EMA or None): EMA object for tracking moving averages or None.
    """

    def __init__(self, config):
        """
        Initializes the Losses class with configuration settings. If LeCam
        regularization is enabled in the configuration, initializes the EMA.

        Args:
            config (dict): Configuration parameters including LeCam settings.
        """
        self.config = config
        self.use_lecam = bool(config["training"]["use_lecam"])
        self.lecam_ratio = float(config["training"]["lecam_ratio"])
        if self.use_lecam and self.lecam_ratio > 0:
            # Initialize EMA object if LeCam regularization is enabled.
            self.ema = EMA(
                int(config["training"]["init_ema"]),
                float(config["training"]["decay_ema"]),
                int(config["training"]["start_epoch_ema"]),
            )
        else:
            self.ema = None

    def lecam_reg(self, real_loss, fake_loss):
        """
        Computes the LeCam Regularization term for GAN training.

        Args:
            real_loss (torch.Tensor): Loss computed from real data.
            fake_loss (torch.Tensor): Loss computed from fake data.
        """
        # Compute the regularization term between real and fake losses
        reg = torch.mean(F.relu(real_loss - self.ema.D_fake).pow(2)) + torch.mean(
            F.relu(self.ema.D_real - fake_loss).pow(2)
        )
        return reg

    def calculate_loss(self, is_discriminator=True, epoch=None):
        """
        Placeholder method to be implemented by subclasses to calculate
        specific losses for discriminators and generators.

        Args:
            is_discriminator (bool): Flag indicating whether to calculate the
                                     loss for the discriminator (True) or
                                     generator (False).
            epoch (int, optional): Current epoch of training, if applicable.
        """
        pass


class EMA:
    """
    Class for managing Exponential Moving Averages (EMA) of losses.

    Attributes:
        G_loss (float): EMA for generator loss.
        D_loss_real (float): EMA for real parts of discriminator loss.
        D_loss_fake (float): EMA for fake parts of discriminator loss.
        D_real (float): EMA for real data outputs.
        D_fake (float): EMA for fake data outputs.
        decay (float): Decay rate for EMA updates.
        start_epoch (int): Epoch from which to start applying EMA.
    """

    def __init__(self, init=100, decay=0.9, start_epoch=0):
        """
        Initializes the EMA class with initial values, decay rate, and
        starting epoch.

        Args:
            init (float): Initial value for EMAs. Default is 100.
            decay (float): Decay rate for updating EMAs. Default is 0.9.
            start_epoch (int): Epoch from which to start applying EMA.
                               Default is 0.

        """
        self.G_loss = init
        self.D_loss_real = init
        self.D_loss_fake = init
        self.D_real = init
        self.D_fake = init
        self.decay = decay
        self.start_epoch = start_epoch

    def update(self, cur, mode, itr):
        """
        Updates the EMA value based on the current loss or output and mode.

        Args:
            cur (float): The current value to update the EMA with.
            mode (str): The mode indicating which EMA to update
                        (G_loss, D_loss_real, D_loss_fake, D_real, D_fake).
            itr (int): The current iteration/epoch of training.
        """
        # Determine decay factor based on the current iteration and start epoch.
        if itr < self.start_epoch:
            decay = 0.0
        else:
            decay = self.decay

        # Update the respective EMA based on the mode.
        if mode == "G_loss":
            self.G_loss = self.G_loss * decay + cur * (1 - decay)
        elif mode == "D_loss_real":
            self.D_loss_real = self.D_loss_real * decay + cur * (1 - decay)
        elif mode == "D_loss_fake":
            self.D_loss_fake = self.D_loss_fake * decay + cur * (1 - decay)
        elif mode == "D_real":
            self.D_real = self.D_real * decay + cur * (1 - decay)
        elif mode == "D_fake":
            self.D_fake = self.D_fake * decay + cur * (1 - decay)
        else:
            # Raise an error if an unrecognized mode is provided.
            raise ValueError(f"Unrecognized mode: {mode}")
