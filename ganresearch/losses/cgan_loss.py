"""
This module defines the CGANLoss class, which extends the Losses class to
calculate the loss for Conditional GANs (CGANs).
"""

import torch
import torch.nn as nn

from ganresearch.losses.losses import Losses


class CGANLoss(Losses):
    """
    CGANLoss class to compute the loss for Conditional GAN models.

    Attributes:
        config (dict): Configuration parameters for the loss function.
    """

    def __init__(self, config):
        """
        Initialize the CGANLoss class with the given configuration.

        Args:
            config (dict): Configuration parameters for initializing the loss
                           function.
        """
        super().__init__(config)

    def calculate_loss(
        self,
        real_output,
        fake_output,
        real_labels,
        fake_labels,
        is_discriminator=True,
        epoch=None,
    ):
        """
        Calculate the Binary Cross-Entropy loss between the model output and
        real labels.

        Args:
            output (torch.Tensor): The output from the generator or
                                   discriminator.
            real_labels (torch.Tensor): The ground truth labels.

        Returns:
            torch.Tensor: The computed loss value.
        """
        # Ensure the inputs are torch.Tensor types

        # Initialize the BCE loss function
        loss_fn = nn.BCELoss()
        if is_discriminator:
            if self.ema is not None:
                # Track the prediction mean for both real and fake outputs
                self.ema.update(torch.mean(fake_output).item(), "D_fake", epoch)
                self.ema.update(torch.mean(real_output).item(), "D_real", epoch)
            # Compute the BCE loss between output and real labels
            d_loss_real = loss_fn(real_output, real_labels)
            g_loss_fake = loss_fn(fake_output, fake_labels)
            if self.use_lecam and self.lecam_ratio > 0 and epoch > self.ema.start_epoch:
                # Apply LeCam regularization if conditions are met
                loss_lecam = self.lecam_reg(d_loss_real, g_loss_fake) * self.lecam_ratio
            else:
                loss_lecam = torch.tensor(0.0)
            return d_loss_real + g_loss_fake + loss_lecam
        else:
            # Compute the BCE loss between output and real labels
            g_loss = loss_fn(fake_output, real_labels)
            return g_loss
