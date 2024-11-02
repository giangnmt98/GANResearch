import datetime
import os
import argparse
from ganresearch.training.dcgan_trainer import DCGANTrainer
from ganresearch.training.cgan_trainer import CGANTrainer
from ganresearch.training.wgan_gp_trainer import WGANGPTrainer
from ganresearch.dataloader.dataloader import DataLoaderManager
from ganresearch.models.gan_factory import GANFactory
from ganresearch.utils.utils import load_config
from ganresearch.evaluation.eval import run_eval
from ganresearch.utils.utils import create_logger

logger = create_logger()


def create_trainer(config, model, train_loader, val_loader, save_path):
    trainers = {
        "cgan": CGANTrainer,
        "dcgan": DCGANTrainer,
        "wgan_gp": WGANGPTrainer
    }
    trainer_class = trainers.get(config["model"]["name"])
    if trainer_class is None:
        raise ValueError(f"Unknown model name: {config['model']['name']}")
    return trainer_class(model=model, config=config, train_loader=train_loader, val_loader=val_loader,
                         save_path=save_path)


def prepare_save_path(config):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(
        config["model"]["save_path"],
        config["model"]["name"],
        config["dataset"]["type"],
        timestamp,
    )
    os.makedirs(save_path, exist_ok=True)
    return save_path


def main(config):
    save_path = prepare_save_path(config)
    logger.info("=" * 50)

    # DataLoader Preparation
    logger.info(f"Preparing DataLoader for {(config['dataset']['type']).upper()} dataset")
    dataloader_manager = DataLoaderManager(config)
    train_loader, val_loader, test_loader = dataloader_manager.get_dataloaders()
    logger.info("=" * 50)

    # Model Creation
    logger.info("Creating GAN model")
    gan_factory = GANFactory(config)
    gan_model = gan_factory.create_model_gan()
    logger.info("=" * 50)

    # Trainer Creation and Training
    trainer = create_trainer(config, gan_model, train_loader, val_loader, save_path)
    logger.info(f"Training {config['model']['name'].upper()} model")
    trainer.train()
    logger.info("=" * 50)

    # Model Evaluation
    logger.info("Evaluating model")
    if config["model"]["name"] == "cgan":
        run_eval(config, gan_model.generator, test_loader, is_loaded=False, save_path=save_path, gen_image=True, has_labels=True)
    else:
        run_eval(config, gan_model.generator, test_loader, is_loaded=False, save_path=save_path, gen_image=True, has_labels=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAN Training and Evaluation")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config file"
    )
    args = parser.parse_args()

    # Read config file
    config = load_config(args.config)

    # Run main program
    main(config)
