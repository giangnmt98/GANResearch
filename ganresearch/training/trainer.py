import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
from torch import nn
from tqdm import tqdm

from ganresearch.training.optimizer import Optimizer
from ganresearch.utils.utils import create_logger

# Initialize the logger with the application name
logger = create_logger()


class Trainer:
    def __init__(self, model, config, train_loader, val_loader, save_path):
        """
        Khởi tạo Trainer với mô hình, cấu hình và DataLoader.

        Args:
            model: Mô hình GAN (ví dụ: DCGAN).
            config: Cấu hình từ file YAML.
            train_loader: DataLoader cho tập huấn luyện.
            val_loader: DataLoader cho tập validation (nếu có).
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
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
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
        """
        # Thêm tqdm vào dataloader để hiển thị tiến độ trong từng epoch
        for i, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch")):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            discriminator.zero_grad()
            real_in_cpu = data[0].to(device)
            batch_size = real_in_cpu.size(0)

            # Train with real
            real_output = discriminator(real_in_cpu)
            d_x = real_output.mean().item()

            # Generate fake images
            noise = torch.randn(
                batch_size,
                self.config["training"]["noise_dimension"],
                1,
                1,
                device=device,
            )
            fake = generator(noise)
            fake_output = discriminator(fake.detach())

            # Calculate discriminator loss
            d_loss = loss_function(
                real_output, fake_output, is_discriminator=True, epoch=i
            )
            d_loss.backward()
            optimizer_d.step()
            d_g_z1 = fake_output.mean().item()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            fake_output = discriminator(fake)

            # Calculate generator loss
            g_loss = loss_function(None, fake_output, is_discriminator=False)
            g_loss.backward()
            optimizer_g.step()

            # Accumulate generator loss
            if ema_losses is not None:
                ema_losses.update(g_loss_total, "G_loss", i)
                g_loss_total += g_loss.item()

            d_g_z2 = fake_output.mean().item()

            # Optional logging
            if i % int(self.config["training"]["log_interval"]) == 0:
                logger.info(
                    "[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f"
                    % (
                        epoch,
                        self.config["training"]["num_epochs"],
                        i,
                        len(dataloader),
                        d_loss.item(),
                        g_loss.item(),
                        d_x,
                        d_g_z1,
                        d_g_z2,
                    )
                )

        return (
            d_loss.item(),
            g_loss.item(),
        )

    def train(self, early_stop=False, patience=5, save_loss=True, gen_images=False):
        """
        Thực hiện huấn luyện mô hình trong số epoch đã chỉ định.
        """

        num_epochs = self.config["training"]["num_epochs"]
        best_loss = float("inf")
        no_improvement_count = 0

        # Khởi tạo trọng số cho generator và discriminator
        self.model.generator.apply(self.weights_init)
        self.model.discriminator.apply(self.weights_init)

        # Tạo optimizer cho generator và discriminator
        optim = Optimizer(self.config)
        self.model.gen_optimizer = optim.create(self.model.generator.parameters())
        self.model.disc_optimizer = optim.create(self.model.discriminator.parameters())

        g_loss_total = 0

        # Tiến trình huấn luyện với thanh progress bar
        for epoch in tqdm(range(1, num_epochs + 1), desc="Training", unit="epoch"):
            gen_loss, disc_loss = self._train_one_epoch(
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

            # Log thông tin về epoch hiện tại
            logger.info(
                f"Epoch [{epoch}/{num_epochs}], "
                f"Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}"
            )

            # Lưu lịch sử loss để vẽ biểu đồ
            self.gen_loss_history.append(gen_loss)
            self.disc_loss_history.append(disc_loss)

            # Early Stopping Logic
            if early_stop:
                avg_loss = (gen_loss + disc_loss) / 2
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    no_improvement_count = 0  # Reset nếu có cải thiện
                else:
                    no_improvement_count += 1
                    logger.info(f"No improvement for {no_improvement_count} epoch(s).")
                    if no_improvement_count >= patience:
                        logger.info("Early stopping triggered.")
                        break

            # Lưu model mỗi save_interval epoch
            if epoch % self.config["training"].get("save_interval", 100) == 0:
                self.save_models(self.save_path, model_name=f"epoch_{epoch}")
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
        self.save_models(self.save_path, model_name="final")
        if save_loss:
            self._save_loss_figure(self.save_path)

    def _save_loss_figure(self, save_path):
        """
        Vẽ và lưu biểu đồ loss của generator và discriminator.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.gen_loss_history, label="Generator Loss")
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
        Lưu các mô hình generator và discriminator vào đĩa.
        """
        torch.save(self.model.generator, f"{save_path}/generator_{model_name}.pth")
        torch.save(
            self.model.discriminator, f"{save_path}/discriminator_{model_name}.pth"
        )
        logger.info(f"Models saved at {save_path}")

    def load_models(self, generator_path=None, discriminator_path=None):
        """
        Tải lại mô hình từ các đường dẫn được chỉ định.

        Args:
            generator_path: Đường dẫn file của generator.
            discriminator_path: Đường dẫn file của discriminator.
        """
        if generator_path:
            self.model.generator.load_state_dict(torch.load(generator_path))
            logger.info(f"Loaded generator from {generator_path}")
        if discriminator_path:
            self.model.discriminator.load_state_dict(torch.load(discriminator_path))
            logger.info(f"Loaded discriminator from {discriminator_path}")
