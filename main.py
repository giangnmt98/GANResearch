import argparse

from ganresearch.training.trainer import Trainer

from ganresearch.dataloader.dataloader import DataLoaderManager
# from evaluation.fid_score import FIDScore
from ganresearch.models.gan_factory import GANFactory
from ganresearch.utils.utils import load_config


def main(config):

    # Chuẩn bị DataLoader
    dataloader_manager = DataLoaderManager(config)
    train_loader, val_loader, _ = dataloader_manager.get_dataloaders()
    # Khởi tạo mô hình
    gan_factory = GANFactory(config)
    gan_model = gan_factory.create_model_gan()

    # Khởi tạo Trainer với mô hình và DataLoader
    trainer = Trainer(gan_model, config, train_loader, val_loader)

    # Bắt đầu huấn luyện với Early Stopping và lưu biểu đồ loss
    trainer.train()

    # # Đánh giá bằng FID Score
    # fid = FIDScore(real_images=[], generated_images=[])
    # fid_score = fid.get_fid_scores()
    # print(f"FID Score: {fid_score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAN Training and Evaluation")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config file"
    )
    args = parser.parse_args()

    # Đọc file config
    config = load_config(args.config)
    # Chạy chương trình chính
    main(config)
