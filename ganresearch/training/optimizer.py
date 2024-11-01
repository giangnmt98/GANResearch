"""
Module for creating and managing optimizers for machine learning models.
"""

import torch.optim as optim


class Optimizer:
    def __init__(self, config):
        """
        Initialize the Optimizer with a given configuration.

        Args:
            config (dict): Configuration containing parameters for the optimizer.
        """
        self.config = config

    def create(self, model_params):
        """
        Create an optimizer based on the algorithm specified in the configuration.

        Args:
            model_params (iterable): Parameters of the model (generator or discriminator).

        Returns:
            torch.optim.Optimizer: Optimizer for the model.

        Raises:
            ValueError: If the optimizer type specified in the config is not supported.
        """
        optimizer_type = self.config["training"].get("optimizer", "adam").lower()

        # Creating Adam optimizer if specified in config
        if optimizer_type == "adam":
            return optim.Adam(
                model_params,
                lr=self.config["training"]["learning_rate"],
                betas=(self.config["training"]["beta1"], 0.999),
            )
        # Creating RMSprop optimizer if specified in config
        elif optimizer_type == "rmsprop":
            return optim.RMSprop(
                model_params, lr=self.config["training"]["learning_rate"]
            )
        # Creating SGD optimizer if specified in config
        elif optimizer_type == "sgd":
            return optim.SGD(
                model_params,
                lr=self.config["training"]["learning_rate"],
                momentum=0.9
            )
        # Creating AdamW optimizer if specified in config
        elif optimizer_type == "adamw":
            return optim.AdamW(
                model_params,
                lr=self.config["training"]["learning_rate"],
                betas=(self.config["training"]["beta1"], 0.999),
            )
        else:
            # Raise error if the optimizer type is not supported
            raise ValueError(f"Optimizer `{optimizer_type}` not supported.")
