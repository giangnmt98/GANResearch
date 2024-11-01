import datetime
import os

import argparse

from ganresearch.training.trainer import Trainer

from ganresearch.dataloader.dataloader import DataLoaderManager
from ganresearch.models.gan_factory import GANFactory
from ganresearch.utils.utils import load_config
from ganresearch.evaluation.eval import run_eval

def main(config):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(
        config["model"]["save_path"],
        config["model"]["name"],
        config["dataset"]["type"],
        timestamp,
    )
    os.makedirs(save_path, exist_ok=True)

    # Chuẩn bị DataLoader
    dataloader_manager = DataLoaderManager(config)
    train_loader, val_loader, test_loader = dataloader_manager.get_dataloaders()

    # Khởi tạo mô hình
    gan_factory = GANFactory(config)

    gan_model = gan_factory.create_model_gan()

    # Khởi tạo Trainer với mô hình và DataLoader
    trainer = Trainer(model=gan_model, config=config, train_loader=train_loader, val_loader=val_loader,save_path=save_path)

    trainer.train()

    run_eval(config, gan_model.generator, test_loader, is_loaded=False, save_path=save_path, gen_image=True)

    # run_eval(config, None, test_loader, is_loaded=True, save_path="models/dcgan/mnist/20241101_233246", gen_image=True)


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
