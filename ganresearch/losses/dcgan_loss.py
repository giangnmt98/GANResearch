"""
This module defines the DCGANLoss class for calculating the loss of a
Deep Convolutional GAN (DCGAN). The loss can be used for both the
discriminator and the generator.
"""

import torch
import torch.nn as nn

from ganresearch.losses.losses import Losses


class DCGANLoss(Losses):
    def __init__(self, config):
        """
        Initialize the DCGANLoss class.

        Args:
            config (dict): Configuration parameters for the loss function.
        """
        super().__init__(config)

    def calculate_loss(
        self, real_output, fake_output, is_discriminator=True, epoch=None
    ):
        """
        Calculate the loss for the DCGAN model.

        Args:
            real_output (torch.Tensor): Discriminator output on real data.
            fake_output (torch.Tensor): Discriminator output on fake data.
            is_discriminator (bool, optional): If True, calculate
                loss for the discriminator. Otherwise, calculate loss
                for the generator. Defaults to True.
            epoch (int, optional): Current epoch number for tracking
                loss (used with EMA). Defaults to None.

        Returns:
            torch.Tensor: Calculated loss.

        Raises:
            ValueError: If EMA is used without specifying 'epoch'.
        """

        # Binary Cross Entropy loss function
        loss_fn = nn.BCELoss()

        if is_discriminator:
            if self.ema is not None:
                # Track the prediction mean for both real and fake outputs
                self.ema.update(torch.mean(fake_output).item(), "D_fake", epoch)
                self.ema.update(torch.mean(real_output).item(), "D_real", epoch)

            # For discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            real_labels = torch.ones_like(real_output)  # Labels for real images
            fake_labels = torch.zeros_like(fake_output)  # Labels for fake images
            real_loss = loss_fn(real_output, real_labels)
            fake_loss = loss_fn(fake_output, fake_labels)

            if self.use_lecam and self.lecam_ratio > 0 and epoch > self.ema.start_epoch:
                # Apply LeCam regularization if conditions are met
                loss_lecam = self.lecam_reg(real_loss, fake_loss) * self.lecam_ratio
            else:
                loss_lecam = torch.tensor(0.0)

            # Total loss for the discriminator
            return real_loss + fake_loss + loss_lecam

        else:
            # For generator: minimize log(1 - D(G(z))) / equivalently maximize log(D(G(z)))
            real_labels = torch.ones_like(fake_output)  # Labels should be real
            return loss_fn(fake_output, real_labels)
