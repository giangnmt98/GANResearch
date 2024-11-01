"""
DCGANTrainer module for training Deep Convolutional Generative Adversarial Networks.
"""

import torch

from ganresearch.training.base_trainer import BaseTrainer
from ganresearch.utils.utils import create_logger

# Initialize the logger with the application name
logger = create_logger()


class DCGANTrainer(BaseTrainer):
    def __init__(self, model, config, train_loader, val_loader, save_path):
        """
        Initialize the DCGANTrainer.

        Parameters:
        model (torch.nn.Module): The GAN model to be trained.
        config (dict): Configuration dictionary.
        train_loader (torch.utils.data.DataLoader): Dataloader for the training set.
        val_loader (torch.utils.data.DataLoader): Dataloader for the validation set.
        save_path (str): Path to save trained models and logs.
        """
        super().__init__(model, config, train_loader, val_loader, save_path)
        self.gen_loss_history = []
        self.disc_loss_history = []

    def _train_one_epoch(
        self,
        epoch: int,
        dataloader: torch.utils.data.DataLoader,
        discriminator: torch.nn.Module,
        generator: torch.nn.Module,
        optimizer_d: torch.optim.Optimizer,
        optimizer_g: torch.optim.Optimizer,
        loss_function: torch.nn.Module,
        ema_losses: any,
        device: torch.device,
        g_loss_total: float,
    ) -> tuple[float, float]:
        """
        Train one epoch of the GAN model.

        Parameters:
        epoch (int): The current epoch number.
        dataloader (torch.utils.data.DataLoader): Dataloader for the training set.
        discriminator (torch.nn.Module): The discriminator model.
        generator (torch.nn.Module): The generator model.
        optimizer_d (torch.optim.Optimizer): Optimizer for the discriminator.
        optimizer_g (torch.optim.Optimizer): Optimizer for the generator.
        loss_function (torch.nn.Module): Loss function.
        ema_losses (any): Exponential moving average of losses.
        device (torch.device): Device to run computations on (CPU or GPU).
        g_loss_total (float): Accumulated generator loss.

        Returns:
        tuple[float, float]: Discriminator and generator loss for the epoch.
        """
        # Add tqdm to dataloader to show progress for each epoch
        for i, data in enumerate(dataloader):
            discriminator.zero_grad()  # Zero the gradients for discriminator
            real_in_cpu = data[0].to(device)  # Move real images to specified device
            batch_size = real_in_cpu.size(0)  # Get the current batch size

            # Train with real images
            real_output = discriminator(
                real_in_cpu
            )  # Discriminator output for real images

            # Generate fake images
            noise = torch.randn(
                batch_size,
                self.config["training"]["noise_dimension"],
                1,
                1,
                device=device,
            )  # Create noise vector
            fake = generator(noise).detach()  # Generate fake images and detach
            fake_output = discriminator(fake)  # Discriminator output for fake images

            # Calculate discriminator loss
            d_loss = loss_function(
                real_output,
                fake_output,
                is_discriminator=True,
                epoch=i,
            )
            d_loss.backward()  # Backpropagate discriminator loss
            optimizer_d.step()  # Update discriminator weights

            generator.zero_grad()  # Zero the gradients for generator

            # Calculate generator loss
            fake_output = discriminator(
                fake
            )  # Get discriminator output for fake images
            g_loss = loss_function(None, fake_output, is_discriminator=False)
            g_loss.backward()  # Backpropagate generator loss
            optimizer_g.step()  # Update generator weights

            # Accumulate generator loss
            if ema_losses is not None:
                ema_losses.update(g_loss_total, "G_loss", i)
                g_loss_total += g_loss.item()

            # Optional logging
            if i % int(self.config["training"]["log_interval"]) == 0:
                logger.info(
                    "[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f"
                    % (
                        epoch,
                        self.config["training"]["num_epochs"],
                        i,
                        len(dataloader),
                        d_loss.item(),
                        g_loss.item(),
                    )
                )
        return d_loss.item(), g_loss.item()
