import os
from PIL import Image, ImageDraw
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

# Simple wrapper that applies EMA to losses.
class ema_losses:
    def __init__(self, init=1000., decay=0.9, start_itr=0):
        self.G_loss = init
        self.D_loss_real = init
        self.D_loss_fake = init
        self.D_real = init
        self.D_fake = init
        self.decay = decay
        self.start_itr = start_itr

    def update(self, cur, mode, itr):
        if itr < self.start_itr:
            decay = 0.0
        else:
            decay = self.decay
        if mode == 'G_loss':
          self.G_loss = self.G_loss*decay + cur*(1 - decay)
        elif mode == 'D_loss_real':
          self.D_loss_real = self.D_loss_real*decay + cur*(1 - decay)
        elif mode == 'D_loss_fake':
          self.D_loss_fake = self.D_loss_fake*decay + cur*(1 - decay)
        elif mode == 'D_real':
          self.D_real = self.D_real*decay + cur*(1 - decay)
        elif mode == 'D_fake':
          self.D_fake = self.D_fake*decay + cur*(1 - decay)


# Simple wrapper that applies EMA to a model. COuld be better done in 1.0 using
# the parameters() and buffers() module functions, but for now this works
# with state_dicts using .copy_
class ema(object):
    def __init__(self, source, target, decay=0.9999, start_itr=0):
        # self.source = source
        # self.target = target
        self.decay = decay
        # Optional parameter indicating what iteration to start the decay at
        self.start_itr = start_itr
        # Initialize target's params to be source's
        # self.source_dict = self.source.state_dict()
        # self.target_dict = self.target.state_dict()
        # print('Initializing EMA parameters to be source parameters...')
        # with torch.no_grad():
        #     for key in self.source_dict:
        #         self.target_dict[key].data.copy_(self.source_dict[key].data)
        #         # target_dict[key].data = source_dict[key].data # Doesn't work!

    def update(self, itr):
        # If an iteration counter is provided and itr is less than the start itr,
        # peg the ema weights to the underlying weights.
        if itr is None:
            decay = self.decay
        elif itr < self.start_itr:#if itr and itr < self.start_itr:
            decay = 0.0
        else:
            decay = self.decay
        # with torch.no_grad():
        #     for key in self.source_dict:
        #         self.target_dict[key].data.copy_(self.target_dict[key].data * decay
        #                                          + self.source_dict[key].data * (1 - decay))



def is_image_file(filename):
    """Check if the file is an image by extension."""
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
                position = ((img.width - text_width) // 2, img.height - text_height - 10)
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
            loop=loop
        )
        print(f"GIF created successfully and saved at {output_gif_path}")
    else:
        print("No valid images found in the folder.")
    # Example usage
    # folder_path = "result"
    # output_gif_path = "output.gif"
    # create_gif_from_folder(folder_path, output_gif_path)
