"""
A module for managing datasets and DataLoaders for GAN research using PyTorch.
"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from ganresearch.datasets import datasets
from ganresearch.utils.utils import create_logger

# Initialize the logger with the application name
logger = create_logger()


class DataLoaderManager:
    def __init__(self, config):
        """
        Initialize DataLoaderManager to manage datasets based on the given config.

        Args:
            config (dict): Configuration dictionary containing dataset and
                           training settings.
        """
        self.config = config
        self.dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.select_dataset()
        self.prepare_dataloaders()

    def select_dataset(self):
        """
        Select dataset based on configuration and initialize it.

        Raises:
            ValueError: If dataset type specified in the config is not supported.
        """
        dataset_type = self.config["dataset"]["type"]
        data_path = self.config["dataset"]["data_path"]
        batch_size = self.config["training"]["batch_size"]

        # Map dataset types to their corresponding classes
        dataset_classes = {
            "mnist": datasets.MNISTDataset,
            "cifar10": datasets.CIFAR10Dataset,
            "cifar100": datasets.CIFAR100Dataset,
            "gtrsb": datasets.GTSRBDataset,
            "flowers102": datasets.Flowers102Dataset,
            "imagenet": datasets.ImageNetDataset,
            "custom": datasets.CustomDataset,
        }

        # Select and initialize the appropriate dataset class
        if dataset_type in dataset_classes:
            self.dataset = dataset_classes[dataset_type](
                config=self.config,
                root=data_path,
                transform=self.get_transform(),
                batch_size=batch_size,
            )
        else:
            raise ValueError(f"Dataset `{dataset_type}` is not supported.")

    def get_transform(self):
        """
        Define and return the transform to be applied to the dataset.

        Returns:
            torchvision.transforms.Compose: Composed transform operations.
        """
        # Define transformations based on the number of channels in the dataset
        if self.config["dataset"]["channels"] == 1:
            return transforms.Compose(
                [
                    transforms.Resize(self.config["dataset"]["image_size"]),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )
        return transforms.Compose(
            [
                transforms.Resize(self.config["dataset"]["image_size"]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def prepare_dataloaders(self):
        """
        Prepare DataLoaders for training, validation, and testing based on config.

        Attributes:
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
            test_loader (torch.utils.data.DataLoader): DataLoader for testing data.
        """
        # Calculate sizes for train, validation, and test datasets
        dataset_size = len(self.dataset.dataset)
        train_size = int(self.config["dataset"]["split_ratio"]["train"] * dataset_size)
        val_size = int(
            self.config["dataset"]["split_ratio"]["validation"] * dataset_size
        )
        test_size = dataset_size - train_size - val_size

        # Split the dataset into train, validation, and test datasets
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            self.dataset.dataset, [train_size, val_size, test_size]
        )

        def custom_collate_fn(batch):
            """
            Custom collate function to handle batch data.

            Args:
                batch (list): Batch data.

            Returns:
                tuple: Processed images and labels.
            """
            transform = transforms.Resize((self.config["dataset"]["image_size"], self.config["dataset"]["image_size"]))
            images, labels = zip(*batch)
            images = [transform(image) for image in images]
            images = torch.stack(images, 0)
            labels = torch.tensor(labels)
            return images, labels

        batch_size = self.config["training"]["batch_size"]  # Batch size

        # Initialize DataLoader for training data
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )

        # Initialize DataLoader for validation data
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )

        # Initialize DataLoader for testing data
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )

    def get_dataloaders(self):
        """
        Get DataLoaders for training, validation, and testing.

        Returns:
            tuple: DataLoaders for training, validation, and testing as
                   (train_loader, val_loader, test_loader).
        """
        return self.train_loader, self.val_loader, self.test_loader
