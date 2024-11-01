"""
This module provides various utilities for image processing and GAN-related tasks.
Includes functions for loading configurations, handling GPUs, validating image files,
generating GIFs, loading GAN generator models, and more.
"""

import os
import matplotlib.pyplot as plt
import torch
import yaml
from PIL import Image, ImageDraw
from ganresearch.utils.custom_logger import CustomLogger

# Initialize the logger with the application name
logger = CustomLogger(name="GAN Research").get_logger()


def load_config(config_path="config.yaml"):
    """
    Load a YAML configuration file and return it as a dictionary.

    Args:
        config_path (str): Path to the YAML configuration file. Default is "config.yaml".

    Returns:
        dict: Loaded configuration as a dictionary.

    Raises:
        FileNotFoundError: If the specified config file does not exist.
        yaml.YAMLError: If the file contains an invalid YAML document.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def create_logger():
    """
    Initialize and retrieve a custom logger instance.

    Returns:
        logging.Logger: Custom logger instance.
    """
    from ganresearch.utils.custom_logger import CustomLogger
    # Initialize the logger with the application name
    logger = CustomLogger(name="GAN Research").get_logger()
    return logger


def get_devices(config):
    """
    Retrieve available devices (GPUs/CPU) based on configuration.

    Args:
        config (dict): Configuration dictionary containing GPU settings.

    Returns:
        list of torch.device: List of available torch devices.

    Raises:
        ValueError: If specified GPU index is invalid.
    """
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


def is_image_file(filename):
    """
    Check if the file is an image by its extension.

    Args:
        filename (str): Name of the file.

    Returns:
        bool: True if the file is an image, otherwise False.
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}
    return os.path.splitext(filename)[1].lower() in image_extensions


def create_gif_from_folder(folder_path, output_gif_path, duration=1000, loop=0):
    """
    Create a GIF from images in a folder, with each frame displaying the image's filename.

    Args:
        folder_path (str): Path to the folder containing images.
        output_gif_path (str): Path to save the generated GIF.
        duration (int): Duration in milliseconds for each frame.
        loop (int): Number of loops for the GIF (0 for infinite).

    Raises:
        IOError: If an image file cannot be opened.
    """
    images = []
    # Sort files to ensure a specific order, if needed
    for filename in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        # Check if the file is an image
        if is_image_file(filename):
            try:
                img = Image.open(file_path).convert("RGB")  # Ensure image is in RGB mode
                draw = ImageDraw.Draw(img)
                # Optional: Load a custom font (requires a .ttf font file)
                # font = ImageFont.truetype("arial.ttf", 20)  # Adjust path and size as needed
                # Draw filename at the bottom of the image
                text = filename
                text_width, text_height = draw.textsize(text)
                position = (
                    (img.width - text_width) // 2,
                    img.height - text_height - 10,
                )
                draw.text(position, text, fill="white")  # Use `font=font` if using a custom font
                images.append(img)
            except IOError:
                print(f"Warning: Could not open image file {file_path}")
    if images:
        # Save as GIF
        images[0].save(
            output_gif_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=loop,
        )
        print(f"GIF created successfully and saved at {output_gif_path}")
    else:
        print("No valid images found in the folder.")


def load_generator(generator_path=None):
    """
    Load a generator model from the specified file path.

    Args:
        generator_path (str, optional): Path to the generator model file.

    Returns:
        torch.nn.Module: Loaded generator model or None if the path is not specified.
    """
    import torch
    if generator_path:
        logger.info(f"Loaded generator from {generator_path}")
        return torch.load(generator_path, weights_only=False)
    else:
        logger.error("Generator path is not specified.")
        return None


def get_class_list_from_dataloader(dataloader):
    """
    Compile a sorted list of unique classes from a DataLoader.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader instance providing the dataset.

    Returns:
        list: Sorted list of unique classes in the dataset.
    """
    class_set = set()
    # Iterate through DataLoader to collect all classes
    for _, labels in dataloader:
        # Add classes from the current batch to the set
        class_set.update(labels.tolist())  # Convert tensor labels to list and add to set
    # Convert the set of classes to a sorted list
    class_list = sorted(list(class_set))
    return class_list


def generate_and_display_images(
        generator_path, list_class, num_images_per_class, noise_dim, save_path
):
    """
    Generate and display images using a GAN generator for given classes.

    Args:
        generator_path (str): Path to the generator model file.
        list_class (list): List of class IDs for which to generate images.
        num_images_per_class (int): Number of images to generate per class.
        noise_dim (int): Dimension of the noise vector for the generator.
        save_path (str): Path to save the generated images.
    """
    generator = load_generator(generator_path)
    all_images = {}
    with torch.no_grad():
        for class_id in list_class:
            images = []
            for _ in range(num_images_per_class):
                # Create a random noise vector
                noise = torch.randn(1, noise_dim, 1, 1)  # (1, noise_dim, 1, 1) for batch size = 1
                generated_image = generator(noise)  # Generate image
                images.append(generated_image.squeeze(0))  # Remove batch dimension
            all_images[class_id] = images
    # Display generated images for each class
    fig, axes = plt.subplots(
        len(list_class),
        num_images_per_class,
        figsize=(num_images_per_class * 2, len(list_class) * 2),
    )
    for class_id, images in all_images.items():
        for i, img in enumerate(images):
            # Convert tensor to numpy for displaying
            img_np = img.permute(1, 2, 0).cpu().numpy()  # Change channels to (H, W, C)
            # Check number of channels and display image accordingly
            if img_np.shape[-1] == 3:  # RGB image
                axes[class_id, i].imshow(img_np)
            else:  # Grayscale image
                axes[class_id, i].imshow(img_np, cmap="gray")
            axes[class_id, i].axis("off")
    # Save the displayed images
    plt.savefig(save_path)
    plt.show()
