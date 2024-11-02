"""
This module is for defining and training GAN models using PyTorch, with functionalities such as
weight initialization, model saving/loading, and logging.
"""

import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
from torch import nn

from ganresearch.evaluation.eval import run_eval_on_train
from ganresearch.training.optimizer import Optimizer
from ganresearch.utils.utils import create_logger

# Initialize the logger with the application name
logger = create_logger()


class BaseTrainer:
    def __init__(self, model, config, train_loader, val_loader, save_path):
        """
        Initialize the Trainer with model, config, and DataLoader.

        Args:
            model: GAN model (e.g., DCGAN).
            config: Configuration from YAML file.
            train_loader: DataLoader for the training set.
            val_loader: DataLoader for the validation set (if available).
            save_path: Directory path where models and logs will be saved.
        """
        self.config = config
        self.model = model
        self.device = config["training"]["device"]
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_function = self.model.losses.calculate_loss
        self.ema_losses = self.model.losses.ema
        self.gen_loss_history = []
        self.disc_loss_history = []
        self.save_path = save_path

    def weights_init(self, m):
        """
        Initialize the weights of the given model layer.

        Args:
            m: A layer in the neural network.
        """
        # Check if the layer is Conv2d or ConvTranspose2d and initialize weights
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        # Check if the layer is BatchNorm2d and initialize weights and biases
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.zeros_(m.bias)

    def _train_one_epoch(
        self,
        epoch,
        dataloader,
        discriminator,
        generator,
        optimizer_d,
        optimizer_g,
        loss_function,
        ema_losses,
        device,
        g_loss_total,
    ):
        """
        Train one epoch of the GAN model.

        Args:
            epoch (int): The current epoch number.
            dataloader (torch.utils.data.DataLoader): DataLoader for the training set.
            discriminator (torch.nn.Module): The discriminator model.
            generator (torch.nn.Module): The generator model.
            optimizer_d (torch.optim.Optimizer): Optimizer for the discriminator.
            optimizer_g (torch.optim.Optimizer): Optimizer for the generator.
            loss_function (torch.nn.Module): The loss function.
            ema_losses (any): Exponential moving average losses.
            device (torch.device): Device to run the model on.
            g_loss_total (float): Total generator loss.

        Returns:
            None
        """
        pass

    def train(self, early_stop=True, patience=5, save_loss=True, gen_images=False):
        """
        Execute the training of the model over a specified number of epochs.

        Args:
            early_stop (bool): Whether to use early stopping.
            patience (int): Number of epochs to wait for improvement before stopping.
            save_loss (bool): Whether to save the loss plot.
            gen_images (bool): Whether to generate and save images during training.

        Returns:
            None
        """
        num_epochs = self.config["training"]["num_epochs"]
        best_fid = float("inf")
        no_improvement_count = 0

        # Initialize weights for generator and discriminator
        self.model.generator.apply(self.weights_init)
        self.model.discriminator.apply(self.weights_init)

        # Create optimizers for generator and discriminator
        optim = Optimizer(self.config)
        self.model.gen_optimizer = optim.create(self.model.generator.parameters())
        self.model.disc_optimizer = optim.create(self.model.discriminator.parameters())

        g_loss_total = 0

        # Training loop with progress bar
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

            # Save loss history for plotting
            self.gen_loss_history.append(gen_loss)
            self.disc_loss_history.append(disc_loss)

            fid_score = run_eval_on_train(
                config=self.config,
                generator=self.model.generator,
                dataloader=self.train_loader,
            )
            logger.info(f"FID Score at Epoch {epoch}: {fid_score:.4f}")

            # Early stopping logic based on FID
            if early_stop:
                if fid_score < best_fid:
                    best_fid = fid_score
                    no_improvement_count = 0  # Reset if improvement occurs
                else:
                    no_improvement_count += 1
                    logger.info(
                        f"No improvement in FID for {no_improvement_count} epoch(s)."
                    )
                    if no_improvement_count >= patience:
                        logger.info("Early stopping triggered due to FID.")
                        break

            # Save model every 'save_interval' epochs
            if epoch % self.config["training"].get("save_interval", 100) == 0:
                if self.config["training"]["save_model_per_epoch"]:
                    self.save_models(self.save_path, model_name=f"epoch_{epoch}")
                if self.config["training"]["gen_images_per_epoch"]:
                    # Generate and save images
                    fake = self.model.generator(
                        torch.randn(
                            self.config["training"]["batch_size"],
                            self.config["training"]["noise_dimension"],
                            1,
                            1,
                            device=self.device,
                        )
                    )
                    vutils.save_image(
                        fake.detach(),
                        "%s/fake_samples_epoch_%03d.png" % (self.save_path, epoch),
                        normalize=True,
                    )

        # Save final model
        self.save_models(self.save_path, model_name="final")

        # Save loss figure if required
        if save_loss:
            self._save_loss_figure(self.save_path)

    def _save_loss_figure(self, save_path):
        """
        Plot and save the loss figure for both generator and discriminator.

        Args:
            save_path (str): Path to save the loss figure.

        Returns:
            None
        """
        plt.figure(figsize=(10, 5))
        # Plot generator loss history
        plt.plot(self.gen_loss_history, label="Generator Loss")
        # Plot discriminator loss history
        plt.plot(self.disc_loss_history, label="Discriminator Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training Loss over Epochs")
        plt.savefig(f"{save_path}/loss.png")
        plt.close()
        logger.info(f"Loss figure saved at {save_path}/loss.png")

    def save_models(self, save_path, model_name):
        """
        Save the generator and discriminator models to disk.

        Args:
            save_path (str): Directory path where models will be saved.
            model_name (str): Name to save the models under.

        Returns:
            None
        """
        torch.save(self.model.generator, f"{save_path}/generator_{model_name}.pth")
        torch.save(
            self.model.discriminator, f"{save_path}/discriminator_{model_name}.pth"
        )
        logger.info(f"Models saved at {save_path}")

    def load_models(self, generator_path=None, discriminator_path=None):
        """
        Load the models from the specified paths.

        Args:
            generator_path (str, optional): Path to the generator model file.
            discriminator_path (str, optional): Path to the discriminator model file.

        Returns:
            None
        """
        if generator_path:
            self.model.generator.load_state_dict(torch.load(generator_path))
            logger.info(f"Loaded generator from {generator_path}")
        if discriminator_path:
            self.model.discriminator.load_state_dict(torch.load(discriminator_path))
            logger.info(f"Loaded discriminator from {discriminator_path}")
