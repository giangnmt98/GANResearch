"""
Module for managing various datasets including MNIST, CIFAR-10, CIFAR-100,
GTSRB, Flowers102, and ImageNet, alongside a custom dataset class.
"""

import os
import random
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets

from ganresearch.utils.utils import create_logger

# Initialize the logger with the application name
logger = create_logger()


class BaseDataLoaderConfig:
    def __init__(self, config, batch_size=32, shuffle=True, transform=None):
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
        self.config = config

    def get_dataloader(self, dataset):
        """
        Returns a DataLoader for the provided dataset.
        Args:
            dataset (Dataset): The dataset to load.
        Returns:
            torch.utils.data.DataLoader: DataLoader instance for the dataset.
        """
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def select_class_id(self, dataset):
        select_class_id = set(self.config["dataset"]["select_class_id"])
        class_imbalance_ratio = self.config["dataset"].get("class_imbalance_ratio", {})
        if select_class_id:
            # Dictionary to hold indices grouped by class
            class_indices = defaultdict(list)

            # Group all indices by their class labels in one loop
            for i, (_, labels) in enumerate(dataset):
                if isinstance(labels, torch.Tensor):
                    for label in labels:  # Iterate through each label in the batch
                        if isinstance(label, torch.Tensor):
                            label = label.item()
                else:
                    label = labels
                if label in select_class_id:
                    class_indices[label].append(i)

            # Create a list to store the final sampled indices
            indices = []

            # Dictionary to store the original and new sample counts for each class
            sample_counts = {}

            for class_id, idx_list in class_indices.items():
                original_count = len(idx_list)
                ratio = class_imbalance_ratio.get(str(class_id), 1.0)
                num_samples = int(original_count * ratio)
                sampled_indices = (
                    random.sample(idx_list, num_samples) if ratio < 1.0 else idx_list
                )
                indices.extend(sampled_indices)

                # Store the counts for later display
                sample_counts[class_id] = {
                    "original": original_count,
                    "selected": len(sampled_indices),
                    "ratio": ratio,
                }

            # Log the sample counts
            logger.info("\nSample Counts by Class:")
            logger.info("Class | Original Samples | Selected Samples | Ratio")
            logger.info("-----------------------------------------------------")
            for class_id, counts in sample_counts.items():
                logger.info(
                    f"{class_id:>5} | {counts['original']:>15} | {counts['selected']:>15} | "
                    f"{counts['ratio']:>5.2f}"
                )

            # Remap class labels to sequential IDs
            sorted_class_ids = sorted(select_class_id)  # Sort to get sequential order
            class_mapping = {
                old_id: new_id for new_id, old_id in enumerate(sorted_class_ids)
            }

            # Create a subset dataset with the sampled indices and remap class labels
            filtered_dataset = Subset(dataset, indices)

            # Apply the new class IDs
            remapped_dataset = [
                (img, class_mapping[label]) for img, label in filtered_dataset
            ]
            return remapped_dataset
        else:
            return dataset

    def __getitem__(self, index):
        image, label = self.data[index]
        if self.transform:
            image = self.transform(image)
        return image, label


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

        dataset = datasets.MNIST(
            root=root, train=train, download=True, transform=self.transform
        )
        return self.select_class_id(dataset)


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
        dataset = datasets.CIFAR10(
            root=root, train=train, download=True, transform=self.transform
        )
        return self.select_class_id(dataset)


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
        dataset = datasets.CIFAR100(
            root=root, train=train, download=True, transform=self.transform
        )
        return self.select_class_id(dataset)


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
        dataset = datasets.GTSRB(root=root, download=True, transform=self.transform)
        return self.select_class_id(dataset)


class CelebADataset(BaseDataLoaderConfig):
    def __init__(self, root="./data", train=True, **kwargs):
        """
        Load CelebA dataset.
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
            datasets.CelebA: The initialized GTSRB dataset.
        """
        dataset = datasets.CelebA(root=root, download=True, transform=self.transform)
        return self.select_class_id(dataset)


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
        dataset = datasets.Flowers102(
            root=root, download=True, transform=self.transform
        )
        return self.select_class_id(dataset)


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
        dataset = datasets.ImageNet(
            root=root, split=split, download=True, transform=self.transform
        )
        return self.select_class_id(dataset)


class StandfordDogsDataset(BaseDataLoaderConfig):
    def __init__(self, config, batch_size, shuffle, transform, data_path):
        super().__init__(config, batch_size, shuffle, transform)
        """
        Initializes the dataset with file path, batch size, and shuffle option.

        Args:
            output_file (str): Path to the .pt file containing the dataset.
            batch_size (int): Batch size for the DataLoader.
            shuffle (bool): Whether to shuffle the data.
        """
        self.data_path = data_path
        self.dataset = None  # Placeholder for loaded dataset
        self.data = None  # Placeholder for loaded data
        self.load_dataset()
        self.dataset = self.initialize_dataset()

    def initialize_dataset(self):
        return self.select_class_id(
            DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        )

    def load_dataset(self):
        """
        Load the dataset from a .pt file.

        Raises:
            FileNotFoundError: If the specified .pt file does not exist.
            ValueError: If the data format is not compatible.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"{self.data_path} does not exist.")

        # Load data from the .pt file
        self.data = torch.load(self.data_path)
        logger.info(f"Dataset loaded from {self.data_path}")

        # Define TensorDataset inline to wrap the loaded data
        class TensorDataset(Dataset):
            def __init__(self, data):
                self.images = data["images"]
                self.labels = data["labels"]

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                return self.images[idx], self.labels[idx]

        # Create and return a DataLoader
        tensor_dataset = TensorDataset(self.data)
        self.dataset = tensor_dataset

        # Verify the structure of the data
        if (
            isinstance(self.data, dict)
            and "images" in self.data
            and "labels" in self.data
        ):
            if not isinstance(self.data["images"], torch.Tensor) or not isinstance(
                self.data["labels"], torch.Tensor
            ):
                raise ValueError(
                    "Data should contain 'images' and 'labels' as torch tensors."
                )
        else:
            raise ValueError(
                "Unsupported data format. Expected a dictionary with 'images' and 'labels' keys."
            )

    def get_dataloader(self):
        """
        Return a DataLoader from the loaded data.

        Returns:
            torch.utils.data.DataLoader: DataLoader for the loaded dataset.
        """
        return self.select_class_id(
            DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        )

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        if self.dataset is None:
            self.load_dataset()
        return len(self.dataset)
