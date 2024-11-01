import os.path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from scipy.linalg import sqrtm
from torchvision import models
from torchvision.models import Inception_V3_Weights

from ganresearch.utils.utils import (create_logger,
                                     get_class_list_from_dataloader,
                                     load_generator)

logger = create_logger()


def create_inception_model(device):
    inception_model = models.inception_v3(
        weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False
    ).to(device)
    inception_model.fc = (
        torch.nn.Identity()
    )  # Replace the FC layer for feature extraction
    return inception_model


def extract_class_features(
    dataloader, generator, inception_model, class_id, noise_dimension, device
):
    """
    Extract features for a specific class from a given data loader using a
    specified model.

    Args:
        class_id (int): Target class ID to extract features for.

    Returns:
        np.ndarray: Extracted features for the specified class.
    """
    inception_model.eval()
    features = []

    # Disable gradient calculation for efficiency
    with torch.no_grad():
        for inputs, labels in dataloader:
            # Filter inputs by the specified class ID
            mask = labels == class_id
            if mask.sum() == 0:
                continue
            inputs = inputs[mask].to(device)

            if generator:
                # Generate fake images using the generator
                noise = torch.randn(
                    inputs.size(0), noise_dimension, 1, 1, device=device
                )
                inputs = generator(noise)

            if inputs.size(1) == 1:  # Kiểm tra nếu ảnh chỉ có 1 kênh
                inputs = inputs.repeat(1, 3, 1, 1)  # Nhân ảnh 1 kênh thành 3 kênh
            # Resize inputs to 299x299 for Inception-v3
            inputs = F.interpolate(
                inputs, size=(299, 299), mode="bilinear", align_corners=False
            )
            outputs = inception_model(inputs)  # Extract features
            features.append(outputs.cpu().numpy())

    return np.concatenate(features, axis=0)


def calculate_fid_score(real_images, generated_images):
    """
    Calculate the Frechet Inception Distance (FID) between the real images and
    generated images. This function follows the formula:
    FID = ||mu_real - mu_fake||^2 + Tr(sigma_real + sigma_fake -
    2 * sqrt(sigma_real * sigma_fake))

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


def calculate_inception_score(generated_images, inception_model, splits=10):
    """
    Calculate the Inception Score for generated images.

    Args:
        generated_images (list): List of generated images.
        splits (int, optional): Number of splits for calculating score. Defaults to 10.

    Returns:
        tuple: A tuple containing the mean and standard deviation of the Inception Score.
    """
    preds = []
    with torch.no_grad():
        for img in generated_images:
            # Ensure image is resized and
            # has correct dimensions for Inception (batch_size, 3, 299, 299)
            if img.dim() == 3:  # If img has shape (3, H, W)
                img = img.unsqueeze(0)  # Add batch dimension
            if img.size(1) == 1:  # Check if the images are grayscale
                img = img.repeat(1, 3, 1, 1)
            # Resize for Inception
            img = F.interpolate(
                img, size=(299, 299), mode="bilinear", align_corners=False
            )
            pred = F.softmax(inception_model(img), dim=1).cpu().numpy()
            preds.append(pred)

    preds = np.concatenate(preds, axis=0)

    # Calculate Inception Score in splits
    split_scores = []
    for k in range(splits):
        part = preds[k * (len(preds) // splits) : (k + 1) * (len(preds) // splits)]
        p_yx = part.mean(axis=0)
        split_score = np.exp(
            np.mean([np.sum(p * (np.log(p) - np.log(p_yx))) for p in part])
        )
        split_scores.append(split_score)

    return np.mean(split_scores), np.std(split_scores)


def run_eval(
    config, eval_generator, dataloader, is_loaded=False, save_path=None, gen_image=False
):
    if not is_loaded:
        generator = eval_generator
    else:
        if save_path is not None:
            generator = load_generator(os.path.join(save_path, "generator.pth"))
        else:
            logger.error("Please provide the path to the saved generator model.")

    noise_dimension = config["training"]["noise_dimension"]
    device = config["training"]["device"]

    inception_model = create_inception_model(device)
    list_scores = []
    list_class = get_class_list_from_dataloader(dataloader)
    for class_idx in list_class:
        print(f"Calculating scores for class {class_idx}...")
        real_features = extract_class_features(
            dataloader, generator, inception_model, class_idx, noise_dimension, device
        )

        fake_features = extract_class_features(
            dataloader, generator, inception_model, class_idx, noise_dimension, device
        )

        fid_score = calculate_fid_score(real_features, fake_features)

        generated_images = [
            generator(torch.randn(1, noise_dimension, 1, 1, device=device))
            for _ in range(100)
        ]

        inception_score, inception_score_std = calculate_inception_score(
            generated_images, inception_model
        )

        # Store FID and IS data for this class
        list_scores.append(
            {"Class": class_idx, "FID Score": fid_score, "IS Score": inception_score}
        )
    df = pd.DataFrame(list_scores)
    df.to_csv(os.path.join(save_path, "scores.csv"), index=False)

    if gen_image:
        noise = torch.randn(
            config["training"]["batch_size"], noise_dimension, 1, 1, device=device
        )
        fake = generator(noise)
        vutils.save_image(
            fake.detach(), os.path.join(save_path, "fake_sample.png"), normalize=True
        )
    return list_scores
