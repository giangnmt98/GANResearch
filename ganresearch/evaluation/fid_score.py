"""
FIDScore module helps in calculating Frechet Inception Distance (FID)
 between real and generated images.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm
from torchvision import models

from ganresearch.utils.custom_logger import CustomLogger

# Initialize the logger with the application name
logger = CustomLogger(name="GAN Research").get_logger()


class FIDScore:
    def __init__(self, real_images, generated_images, device):
        """
        Initialize FIDScore evaluator.

        Args:
            real_images (np.ndarray): Features of real images.
            generated_images (np.ndarray): Features of generated images.
            device (torch.device): Device to perform computations on.
        """
        self.real_images = real_images
        self.generated_images = generated_images
        self.device = device

    @staticmethod
    def calculate_fid(real_images, generated_images):
        """
        Calculate the Frechet Inception Distance (FID) between the real images and
        generated images. This function follows the formula:
        FID = ||mu_real - mu_fake||^2 + Tr(sigma_real + sigma_fake -
        2 * sqrt(sigma_real * sigma_fake))

        Args:
            real_images (np.ndarray): Numpy array of features from real images.
            generated_images (np.ndarray): Numpy array of
            features from generated images.

        Returns:
            float: The calculated FID score.
        """
        # Compute mean and covariance for real images
        mu_real = real_images.mean(axis=0)
        sigma_real = np.cov(real_images, rowvar=False)

        # Compute mean and covariance for generated images
        mu_fake = generated_images.mean(axis=0)
        sigma_fake = np.cov(generated_images, rowvar=False)

        # Compute the matrix square root of the product of covariance matrices
        covmean = sqrtm(sigma_real @ sigma_fake)

        # Check for complex numbers in the result and handle them
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        # Calculate the FID score using the formula
        fid = np.sum((mu_real - mu_fake) ** 2) + np.trace(
            sigma_real + sigma_fake - 2 * covmean
        )
        return fid

    def extract_class_features(self, loader, model, class_id, generator=None):
        """
        Extract features for a specific class from a given data loader using a
        specified model.

        Args:
            loader (torch.utils.data.DataLoader): Data loader for the dataset.
            model (torch.nn.Module): Model used for feature extraction.
            class_id (int): Target class ID to extract features for.
            generator (torch.nn.Module, optional): Optional generator model for
            generating images.

        Returns:
            np.ndarray: Extracted features for the specified class.
        """
        model.eval()
        features = []

        # Disable gradient calculation for efficiency
        with torch.no_grad():
            for inputs, labels in loader:
                # Filter inputs by the specified class ID
                mask = labels == class_id
                if mask.sum() == 0:
                    continue
                inputs = inputs[mask].to(self.device)

                if generator:
                    # Generate fake images using the generator
                    noise = torch.randn(inputs.size(0), 100, 1, 1, device=self.device)
                    inputs = generator(noise)

                # Resize inputs to 299x299 for Inception-v3
                inputs = F.interpolate(
                    inputs, size=(299, 299), mode="bilinear", align_corners=False
                )
                outputs = model(inputs)  # Extract features
                features.append(outputs.cpu().numpy())

        return np.concatenate(features, axis=0)

    def get_fid_scores(self, test_loader, generator, number_class: int):
        """
        Compare FID scores for each class between balanced and imbalanced generators.

        Args:
            test_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
            generator (torch.nn.Module): Generator model.
            number_class (int): Number of classes to evaluate.

        Returns:
            list of dict: List containing FID scores
             and their differences for each class.
        """
        # Initialize the Inception-v3 model for feature extraction
        inception = models.inception_v3(pretrained=True, transform_input=False).to(
            self.device
        )
        inception.fc = torch.nn.Identity()  # Replace the FC layer

        fid_data = []

        for class_id in range(number_class):
            logger.info(f"Processing class {class_id}...")

            # Extract real features for the current class
            real_images = self.extract_class_features(test_loader, inception, class_id)

            # Extract fake features from the generator
            generated_images = self.extract_class_features(
                test_loader, inception, class_id, generator
            )

            # Calculate FID score
            fid_score = self.calculate_fid(real_images, generated_images)

            # Store results for the current class
            fid_data.append(
                {
                    "Class": class_id,
                    "FID": fid_score,
                }
            )

        return fid_data
