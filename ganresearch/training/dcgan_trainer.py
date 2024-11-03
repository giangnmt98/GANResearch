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
        real_label=1,
        fake_label=0,
        criterion=None,
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
            discriminator.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full(
                (batch_size,), real_label, dtype=real_cpu.dtype, device=device
            )

            real_output = discriminator(real_cpu)
            real_loss = criterion(real_output, label)
            real_loss.backward()

            # train with fake
            noise = torch.randn(
                batch_size,
                self.config["training"]["noise_dimension"],
                1,
                1,
                device=device,
            )
            fake = generator(noise)
            label.fill_(fake_label)
            fake_output = discriminator(fake.detach())
            fake_loss = criterion(fake_output, label)
            fake_loss.backward()
            d_loss = loss_function(
                real_loss, fake_loss, real_output, fake_output, epoch=i
            )
            optimizer_d.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = discriminator(fake)
            g_loss = criterion(output, label)
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
        return d_loss.item(), g_loss.item()
