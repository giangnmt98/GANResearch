"""
WGANGPTrainer module for training GAN models with Wasserstein loss and Gradient Penalty.
"""

import torch

from ganresearch.training.base_trainer import BaseTrainer
from ganresearch.utils.utils import create_logger

# Initialize the logger with the application name
logger = create_logger()


class WGANGPTrainer(BaseTrainer):
    def __init__(self, model, config, train_loader, val_loader, save_path):
        """
        Initialize the WGANGPTrainer class.

        Args:
            model (torch.nn.Module): The GAN model to be trained.
            config (dict): Configuration dictionary.
            train_loader (torch.utils.data.DataLoader): DataLoader for training set.
            val_loader (torch.utils.data.DataLoader): DataLoader for validation set.
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

        Args:
            epoch (int): The current training epoch.
            dataloader (torch.utils.data.DataLoader): DataLoader for the training set.
            discriminator (torch.nn.Module): The discriminator model.
            generator (torch.nn.Module): The generator model.
            optimizer_d (torch.optim.Optimizer): Optimizer for the discriminator.
            optimizer_g (torch.optim.Optimizer): Optimizer for the generator.
            loss_function (torch.nn.Module): Loss function for the GAN.
            ema_losses (any): Exponential moving average of losses.
            device (torch.device): Device to run the models on.
            g_loss_total (float): Cumulative generator loss.

        Returns:
            tuple[float, float]: Discriminator loss and generator loss for the epoch.

        Raises:
            ValueError: If an error occurs in the loss function or training steps.
        """
        # Add tqdm to dataloader to show progress for each epoch
        for i, data in enumerate(dataloader):
            # Zero the parameter gradients for the discriminator
            discriminator.zero_grad()
            real_in_cpu = data[0].to(device)
            batch_size = real_in_cpu.size(0)

            # Train with real data
            real_output = discriminator(real_in_cpu).mean()

            # Generate fake images
            noise = torch.randn(
                batch_size,
                self.config["training"]["noise_dimension"],
                1,
                1,
                device=device,
            )
            fake = generator(noise).detach()
            fake_output = discriminator(fake).mean()

            # Calculate discriminator loss
            d_loss = loss_function(
                real_output,
                fake_output,
                is_discriminator=True,
                epoch=i,
                critic=discriminator,
                real_data=real_in_cpu,
                fake_data=fake,
                device=self.device,
            )
            d_loss.backward()
            optimizer_d.step()

            # Update Generator every n_critic iterations
            if i % self.config["training"]["n_critic"] == 0:
                generator.zero_grad()
                fake_data = generator(noise)

                # Calculate generator loss
                g_loss = -discriminator(fake_data).mean()
                g_loss.backward()
                optimizer_g.step()

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

        return (d_loss.item(), g_loss.item())
