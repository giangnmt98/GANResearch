"""
This module defines the CGANLoss class, which extends the Losses class to
calculate the loss for Conditional GANs (CGANs).
"""

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

    def calculate_loss(self, output, real_labels):
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

        # Compute the BCE loss between output and real labels
        return loss_fn(output, real_labels)
