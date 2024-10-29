import datetime
import os

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch import nn

from ganresearch.training.optimizer import Optimizer
from ganresearch.utils.utils import create_logger

# Initialize the logger with the application name
logger = create_logger()


class Trainer:
    def __init__(self, model, config, train_loader, val_loader=None):
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
        self.gen_loss_history = []
        self.disc_loss_history = []

    def weights_init(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.zeros_(m.bias)

    from tqdm import tqdm  # Import tqdm for progress bar

    def train(self, early_stop=False, patience=5, save_loss=True):
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

        # Tiến trình huấn luyện với thanh progress bar
        for epoch in tqdm(range(1, num_epochs + 1), desc="Training", unit="epoch"):
            gen_loss, disc_loss = self._train_one_epoch()

            # Log thông tin về epoch hiện tại
            logger.info(
                f"Epoch [{epoch}/{num_epochs}], "
                f"Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}"
            )

            # Lưu lịch sử loss để vẽ biểu đồ
            self.gen_loss_history.append(gen_loss)
            self.disc_loss_history.append(disc_loss)

            # Hiển thị loss mỗi vài epoch (tùy theo config)
            if epoch % self.config["training"].get("display_interval", 10) == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}] | Gen Loss: {gen_loss:.4f} | "
                    f"Disc Loss: {disc_loss:.4f}"
                )

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
                self.save_models()

        # Vẽ và lưu biểu đồ loss nếu được yêu cầu
        if save_loss:
            self._save_loss_figure()

    def _train_one_epoch(self):
        """
        Huấn luyện một epoch với logic từ DCGAN.

        Args:
            epoch: Số thứ tự của epoch hiện tại.

        Returns:
            avg_gen_loss (float): Giá trị loss trung bình của generator.
            avg_disc_loss (float): Giá trị loss trung bình của discriminator.
        """
        self.model.generator.train()
        self.model.discriminator.train()

        gen_loss_sum = 0.0
        disc_loss_sum = 0.0

        for real_images, _ in self.train_loader:
            real_images = real_images.to(self.device)
            batch_size = real_images.size(0)

            # Sinh noise ngẫu nhiên cho generator
            noise = torch.randn(
                batch_size,
                self.config["training"]["noise_dimension"],
                1,
                1,
                device=self.device,
            )

            # Huấn luyện discriminator
            self.model.discriminator.zero_grad()
            real_output = self.model.discriminator(real_images)
            fake_images = self.model.generator(noise).detach()
            fake_output = self.model.discriminator(fake_images)

            disc_loss = self.model.loss(real_output, fake_output)
            disc_loss.backward()
            self.model.disc_optimizer.step()

            # Huấn luyện generator
            self.model.generator.zero_grad()
            fake_images = self.model.generator(noise)
            fake_output = self.model.discriminator(fake_images)

            gen_loss = self.model.criterion(fake_output, torch.ones_like(fake_output))
            gen_loss.backward()
            self.model.gen_optimizer.step()

            # Cộng dồn loss
            gen_loss_sum += gen_loss.item()
            disc_loss_sum += disc_loss.item()

        # Tính loss trung bình cho epoch
        avg_gen_loss = gen_loss_sum / len(self.train_loader)
        avg_disc_loss = disc_loss_sum / len(self.train_loader)
        return avg_gen_loss, avg_disc_loss

    def _save_loss_figure(self):
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

        save_path = self.config["gan_model"]["save_model_path"]
        os.makedirs(save_path, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{save_path}/{timestamp}_loss.png")
        plt.close()
        logger.info(f"Loss figure saved at {save_path}/{timestamp}_loss.png")

    def save_models(self):
        """
        Lưu các mô hình generator và discriminator vào đĩa.
        """
        save_path = self.config["gan_model"]["save_model_path"]
        os.makedirs(save_path, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(
            self.model.generator.state_dict(), f"{save_path}/{timestamp}_generator.pth"
        )
        torch.save(
            self.model.discriminator.state_dict(),
            f"{save_path}/{timestamp}_discriminator.pth",
        )
        logger.info(f"Models saved at {save_path}/{timestamp}_generator.pth")

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
