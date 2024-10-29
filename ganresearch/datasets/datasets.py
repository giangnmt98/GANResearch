"""
Module for managing various datasets including MNIST, CIFAR-10, CIFAR-100,
GTSRB, Flowers102, and ImageNet, alongside a custom dataset class.
"""

import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from tqdm import tqdm

from ganresearch.utils.utils import create_logger

# Initialize the logger with the application name
logger = create_logger()


class BaseDataLoaderConfig:
    def __init__(self, batch_size=32, shuffle=True, transform=None):
        """
        Basic class containing common attributes and methods.
        Args:
            batch_size (int): Size of each batch for DataLoader.
            shuffle (bool): Shuffle the data in DataLoader.
            transform (callable): Transformations applied to images.
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform

    def get_dataloader(self, dataset):
        """
        Returns a DataLoader for the provided dataset.
        Args:
            dataset (Dataset): The dataset to load.
        Returns:
            torch.utils.data.DataLoader: DataLoader instance for the dataset.
        """
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)


class MNISTDataset(BaseDataLoaderConfig):
    def __init__(self, root="./data", train=True, **kwargs):
        """
        Load MNIST dataset.
        Args:
            root (str): Root directory of dataset.
            train (bool): Whether to use training or test split.
        """
        super().__init__(**kwargs)
        self.dataset = self.initialize_dataset(root, train)

    def initialize_dataset(self, root, train):
        """
        Initialize the MNIST dataset.
        Args:
            root (str): Root directory of dataset.
            train (bool): Whether to use training or test split.
        Returns:
            datasets.MNIST: The initialized MNIST dataset.
        """
        return datasets.MNIST(
            root=root, train=train, download=True, transform=self.transform
        )


class CIFAR10Dataset(BaseDataLoaderConfig):
    def __init__(self, root="./data", train=True, **kwargs):
        """
        Load CIFAR-10 dataset.
        Args:
            root (str): Root directory of dataset.
            train (bool): Whether to use training or test split.
        """
        super().__init__(**kwargs)
        self.dataset = self.initialize_dataset(root, train)

    def initialize_dataset(self, root, train):
        """
        Initialize the CIFAR-10 dataset.
        Args:
            root (str): Root directory of dataset.
            train (bool): Whether to use training or test split.
        Returns:
            datasets.CIFAR10: The initialized CIFAR-10 dataset.
        """
        return datasets.CIFAR10(
            root=root, train=train, download=True, transform=self.transform
        )


class CIFAR100Dataset(BaseDataLoaderConfig):
    def __init__(self, root="./data", train=True, **kwargs):
        """
        Load CIFAR-100 dataset.
        Args:
            root (str): Root directory of dataset.
            train (bool): Whether to use training or test split.
        """
        super().__init__(**kwargs)
        self.dataset = self.initialize_dataset(root, train)

    def initialize_dataset(self, root, train):
        """
        Initialize the CIFAR-100 dataset.
        Args:
            root (str): Root directory of dataset.
            train (bool): Whether to use training or test split.
        Returns:
            datasets.CIFAR100: The initialized CIFAR-100 dataset.
        """
        return datasets.CIFAR100(
            root=root, train=train, download=True, transform=self.transform
        )


class GTSRBDataset(BaseDataLoaderConfig):
    def __init__(self, root="./data", train=True, **kwargs):
        """
        Load GTSRB dataset.
        Args:
            root (str): Root directory of dataset.
            train (bool): Whether to use training or test split.
        """
        super().__init__(**kwargs)
        self.dataset = self.initialize_dataset(root, train)

    def initialize_dataset(self, root, train):
        """
        Initialize the GTSRB dataset.
        Args:
            root (str): Root directory of dataset.
            train (bool): Whether to use training or test split.
        Returns:
            datasets.GTSRB: The initialized GTSRB dataset.
        """
        return datasets.GTSRB(root=root, train=train, transform=self.transform)


class Flowers102Dataset(BaseDataLoaderConfig):
    def __init__(self, root="./data", train=True, **kwargs):
        """
        Load Flowers102 dataset.
        Args:
            root (str): Root directory of dataset.
            train (bool): Whether to use training or test split.
        """
        super().__init__(**kwargs)
        self.dataset = self.initialize_dataset(root, train)

    def initialize_dataset(self, root, train):
        """
        Initialize the Flowers102 dataset.
        Args:
            root (str): Root directory of dataset.
            train (bool): Whether to use training or test split.
        Returns:
            datasets.Flowers102: The initialized Flowers102 dataset.
        """
        return datasets.Flowers102(root=root, train=train, transform=self.transform)


class ImageNetDataset(BaseDataLoaderConfig):
    def __init__(self, root="./data", split="train", **kwargs):
        """
        Load ImageNet dataset.
        Args:
            root (str): Root directory of dataset.
            split (str): Whether to use 'train' or 'val' split.
        """
        super().__init__(**kwargs)
        self.dataset = self.initialize_dataset(root, split)

    def initialize_dataset(self, root, split):
        """
        Initialize the ImageNet dataset.
        Args:
            root (str): Root directory of dataset.
            split (str): Whether to use 'train' or 'val' split.
        Returns:
            datasets.ImageNet: The initialized ImageNet dataset.
        """
        return datasets.ImageNet(root=root, split=split, transform=self.transform)


class CustomDataset(BaseDataLoaderConfig):
    def __init__(self, root_dir, output_file="dataset.pt", **kwargs):
        """
        Initialize CustomDataset with the given attributes.

        Args:
            root_dir (str): Path to the directory containing images and classes.
            output_file (str): Name of the .pt file to save the dataset.
        """
        super().__init__(**kwargs)
        self.root_dir = root_dir
        self.output_file = output_file
        self.data = None

    def is_image_file(self, filename):
        """
        Check if a file is an image.

        Args:
            filename (str): The filename to check.

        Returns:
            bool: True if the file is an image, False otherwise.
        """
        IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]
        return any(filename.lower().endswith(ext) for ext in IMG_EXTENSIONS)

    def create_and_save_dataset(self):
        """
        Create the dataset from a directory and save it to a .pt file.
        """
        data = []  # List to store (image, label) tuples
        classes = sorted(os.listdir(self.root_dir))
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        total_files = sum(len(files) for _, _, files in os.walk(self.root_dir))

        # Initialize progress bar
        progress_bar = tqdm(total=total_files, desc="Processing images", unit="file")

        # Loop through subdirectories to gather data
        for class_name in classes:
            class_path = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            for img_name in os.listdir(class_path):
                if self.is_image_file(img_name):
                    img_path = os.path.join(class_path, img_name)
                    image = Image.open(img_path).convert("RGB")
                    image = self.transform(image)
                    label = class_to_idx[class_name]
                    data.append((image, label))
                    progress_bar.update(1)  # Update progress bar

        progress_bar.close()  # Close progress bar

        # Convert image list and labels into Tensor
        images, labels = zip(*data)
        dataset = {
            "images": torch.stack(images),
            "labels": torch.tensor(labels),
            "class_to_idx": class_to_idx,  # Include class-to-idx mapping
        }
        # Check if the file already exists before saving
        if os.path.exists(self.output_file):
            logger.info(
                f"File {self.output_file} already exists. Overwriting the file."
            )
        else:
            logger.info(f"File {self.output_file} does not exist. Creating a new file.")

        # Save the dataset to a .pt file
        torch.save(dataset, self.output_file)
        logger.info(f"Dataset saved to {self.output_file}")

    def load_dataset(self):
        """
        Load the dataset from a .pt file.

        Raises:
            FileNotFoundError: If the specified .pt file does not exist.
        """
        if not os.path.exists(self.output_file):
            raise FileNotFoundError(f"{self.output_file} does not exist.")
        self.data = torch.load(self.output_file)
        logger.info(f"Dataset loaded from {self.output_file}")

    def get_dataloader(self, batch_size=32, shuffle=True):
        """
        Override to return a DataLoader from the loaded data.

        Args:
            batch_size (int): Size of each batch.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            torch.utils.data.DataLoader: DataLoader for the loaded dataset.
        """
        if self.data is None:
            self.load_dataset()

        class TensorDataset(Dataset):
            def __init__(self, data):
                self.images = data["images"]
                self.labels = data["labels"]

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                return self.images[idx], self.labels[idx]

        tensor_dataset = TensorDataset(self.data)
        return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)


# -------------------------
# Sử dụng các class dataset
# -------------------------

# # Define the transformation with single-channel image normalization
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
#     # Normalize the single channel image
#     transforms.Normalize(mean=[0.5], std=[0.5])
# ])
#
# # Load the MNIST dataset with the transformation
# mnist_dataset = datasets.MNIST(root='./data', train=True,
# transform=transform, download=True)
# mnist_loader = torch.utils.data.DataLoader(mnist_dataset,
# batch_size=64, shuffle=True)
#
# # Duyệt qua 1 batch dữ liệu MNIST
# for images, labels in mnist_loader:
#     logger.info(f"Batch size: {images.size()}, Labels: {labels}")
#     break

# cifar10_dataset = CIFAR10Dataset()
# cifar10_loader = cifar10_dataset.get_dataloader()
#
# # Duyệt qua 1 batch dữ liệu MNIST
# for images, labels in cifar10_loader:
#     logger.info(f"Batch size: {images.size()}, Labels: {labels}")
#     break

# -------------------------
# Sử dụng CustomDataset
# -------------------------

# # Khởi tạo CustomDataset và tạo DataLoader
# custom_dataset = CustomDataset(root_dir='./data/Stanford Dogs Dataset',
# output_file='./data/StanfordDogsDataset.pt')
#
# # Tạo và lưu dataset
# custom_dataset.create_and_save_dataset()
#
# # Load dataset và duyệt qua DataLoader
# dataloader = custom_dataset.get_dataloader(batch_size=32)
#
# for images, labels in dataloader:
#     logger.info(f'Batch size: {images.size()}, Labels: {labels}')
#     break
