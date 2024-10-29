import torch
import yaml

from ganresearch.utils.custom_logger import CustomLogger

# Initialize the logger with the application name
logger = CustomLogger(name="GAN Research").get_logger()


def load_config(config_path="config.yaml"):
    """
    Load file config YAML và trả về dưới dạng dict.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def create_logger():
    from ganresearch.utils.custom_logger import CustomLogger

    # Initialize the logger with the application name
    logger = CustomLogger(name="GAN Research").get_logger()
    return logger


def get_devices(config):
    gpu_indices = list(config["training"]["list_gpus"])

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        devices = []
        for idx in gpu_indices:
            if idx >= num_gpus:
                raise ValueError(
                    f"Invalid GPU index: {idx}. Available GPUs: {num_gpus}"
                )
            devices.append(torch.device(f"cuda:{idx}"))
            logger.info(f"Using GPU: {torch.cuda.get_device_name(idx)} (GPU {idx})")
    else:
        devices = [torch.device("cpu")]
        logger.info("Using CPU")

    return devices
