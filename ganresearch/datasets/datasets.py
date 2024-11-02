"""
Module for managing various datasets including MNIST, CIFAR-10, CIFAR-100,
GTSRB, Flowers102, and ImageNet, alongside a custom dataset class.
"""

import os
import random
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from tqdm import tqdm

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
            for i, (_, label) in enumerate(dataset):
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

# Khởi tạo CustomDataset và tạo DataLoader
# custom_dataset = CustomDataset(root_dir='./data/Stanford Dogs Dataset',
# output_file='./data/StanfordDogsDataset.pt')
#
# # Tạo và lưu dataset
# custom_dataset.create_and_save_dataset()

# Load dataset và duyệt qua DataLoader
# dataloader = custom_dataset.get_dataloader(batch_size=32)
#
# for images, labels in dataloader:
#     logger.info(f'Batch size: {images.size()}, Labels: {labels}')
#     break
