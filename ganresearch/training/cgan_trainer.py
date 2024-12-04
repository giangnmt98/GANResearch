"""
Module for training Conditional Generative Adversarial Networks (CGAN) using PyTorch.
"""

import torch
import torchvision.utils as vutils

from ganresearch.evaluation.eval import run_eval_on_train
from ganresearch.training.base_trainer import BaseTrainer
from ganresearch.training.optimizer import Optimizer
from ganresearch.utils.utils import create_logger

# Initialize the logger with the application name
logger = create_logger()


class CGANTrainer(BaseTrainer):
    """
    Trainer class for Conditional GANs (CGAN). Inherits from BaseTrainer.

    Args:
        model (torch.nn.Module): The GAN model to be trained.
        config (dict): Configuration dictionary.
        train_loader (torch.utils.data.DataLoader): Dataloader for the training set.
        val_loader (torch.utils.data.DataLoader): Dataloader for the validation set.
        save_path (str): Path to save trained models and logs.
    """

    def __init__(self, model, config, train_loader, val_loader, save_path):
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
            epoch (int): Current epoch number.
            dataloader (torch.utils.data.DataLoader): Dataloader for training data.
            discriminator (torch.nn.Module): The discriminator network.
            generator (torch.nn.Module): The generator network.
            optimizer_d (torch.optim.Optimizer): Optimizer for the discriminator.
            optimizer_g (torch.optim.Optimizer): Optimizer for the generator.
            loss_function (torch.nn.Module): Loss function for training.
            ema_losses (any): EMA (Exponential Moving Average) of losses.
            device (torch.device): Computation device (CPU/GPU).
            g_loss_total (float): Cumulative generator loss.

        Returns:
            tuple[float, float]: Discriminator loss and generator loss for the epoch.

        Raises:
            No specific exceptions are raised.
        """
        # Add tqdm to dataloader to show progress for each epoch
        for i, (real_images, labels) in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            labels = labels.to(device)

            # Train Discriminator
            discriminator.zero_grad()  # Zero the gradients
            real_labels = torch.full(
                (batch_size,), 1, dtype=torch.float, device=device
            )  # Real labels
            fake_labels = torch.full(
                (batch_size,), 0, dtype=torch.float, device=device
            )  # Fake labels
            # Discriminator output on real images
            real_output = discriminator(real_images, labels).view(-1)

            # Generate noise for fake images
            noise = torch.randn(
                batch_size,
                self.config["training"]["noise_dimension"],
                1,
                1,
                device=device,
            )
            fake_images = generator(
                noise, labels
            )  # Use generator to create fake images
            # Discriminator output on fake images
            fake_output = discriminator(fake_images.detach(), labels).view(-1)

            d_loss = loss_function(
                real_output,
                fake_output,
                real_labels,
                fake_labels,
                is_discriminator=True,
                epoch=i,
            )  # Compute loss for fake images

            d_loss.backward()  # Backpropagate the loss for real images
            optimizer_d.step()  # Update discriminator weights

            # Train Generator
            generator.zero_grad()  # Zero the gradients
            fake_output = discriminator(fake_images, labels).view(
                -1
            )  # Discriminator output on fake images
            g_loss = loss_function(
                real_output,
                fake_output,
                real_labels,
                fake_labels,
                is_discriminator=False,
                epoch=i,
            )  # Compute loss for generator
            g_loss.backward()  # Backpropagate the generator loss
            optimizer_g.step()  # Update generator weights

            # Accumulate generator loss
            if ema_losses is not None:
                ema_losses.update(g_loss_total, "G_loss", i)
                g_loss_total += g_loss.item()

            # Optional logging
            if i % int(self.config["training"]["log_interval"]) == 0:
                logger.info(
                    "[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f",
                    epoch,
                    self.config["training"]["num_epochs"],
                    i,
                    len(dataloader),
                    d_loss.item(),
                    g_loss.item(),
                )
        return d_loss.item(), g_loss.item()

    def train(self, early_stop=True, patience=5, save_loss=True, gen_images=False):
        """
        Train the GAN model over a specified number of epochs.

        Args:
            early_stop (bool, optional): Whether to enable early stopping. Defaults to False.
            patience (int, optional): Number of epochs to wait for
             improvement before stopping if early_stop is True. Defaults to 5.
            save_loss (bool, optional): Whether to save the loss figure. Defaults to True.
            gen_images (bool, optional): Whether to generate and
             save images during training. Defaults to False.
        """
        num_epochs = self.config["training"]["num_epochs"]
        best_fid = float("inf")
        no_improvement_count = 0

        # Initialize weights for generator and discriminator
        self.model.generator.apply(self.weights_init)
        self.model.discriminator.apply(self.weights_init)

        # Create optimizer for generator and discriminator
        optim = Optimizer(self.config)
        self.model.gen_optimizer = optim.create(self.model.generator.parameters())
        self.model.disc_optimizer = optim.create(self.model.discriminator.parameters())

        g_loss_total = 0
        # Progress bar for training process
        for epoch in range(1, num_epochs + 1):
            disc_loss, gen_loss = self._train_one_epoch(
                epoch,
                self.train_loader,
                self.model.discriminator,
                self.model.generator,
                self.model.disc_optimizer,
                self.model.gen_optimizer,
                self.loss_function,
                self.ema_losses,
                self.device,
                g_loss_total,
            )

            # Log information about the current epoch
            logger.info("=" * 50)
            logger.info(
                f"Epoch [{epoch}/{num_epochs}], "
                f"Disc Loss: {disc_loss:.4f}, Gen Loss: {gen_loss:.4f}"
            )

            # Append loss history for plotting
            self.gen_loss_history.append(gen_loss)
            self.disc_loss_history.append(disc_loss)

            # # Early stopping logic based on FID
            # if early_stop:
            #     fid_score = run_eval_on_train(
            #         config=self.config,
            #         generator=self.model.generator,
            #         dataloader=self.val_loader,
            #         has_labels=True,
            #     )
            #     logger.info(f"FID Score at Epoch {epoch}: {fid_score:.4f}")
            #     if fid_score < best_fid:
            #         best_fid = fid_score
            #         no_improvement_count = 0  # Reset if improvement occurs
            #     else:
            #         no_improvement_count += 1
            #         logger.info(
            #             f"No improvement in FID for {no_improvement_count} epoch(s)."
            #         )
            #         if no_improvement_count >= patience:
            #             logger.info("Early stopping triggered due to FID.")
            #             break

            # Save model at each 'save_interval' epoch
            if epoch % self.config["training"].get("save_interval", 100) == 0:
                if self.config["training"]["save_model_per_epoch"]:
                    self.save_models(self.save_path, model_name=f"epoch_{epoch}")
                if self.config["training"]["gen_images_per_epoch"]:
                    labels = torch.randint(
                        0,
                        10,
                        (self.config["training"]["batch_size"],),
                        device=self.device,
                    )
                    fake = self.model.generator(
                        torch.randn(
                            self.config["training"]["batch_size"],
                            self.config["training"]["noise_dimension"],
                            1,
                            1,
                            device=self.device,
                        ),
                        labels,
                    )
                    vutils.save_image(
                        fake.detach(),
                        "%s/fake_samples_epoch_%03d.png" % (self.save_path, epoch),
                        normalize=True,
                    )

        # Save final model after training is completed
        self.save_models(self.save_path, model_name="final")

        # Save loss figure if required
        if save_loss:
            self._save_loss_figure(self.save_path)
