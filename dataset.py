import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class FoodTrainDataset(Dataset):
    """
    PyTorch Dataset class for training/validation images and their multi-label annotations.

    Expects:
    - A CSV file with columns: [Filename, 0, 1, 2, ..., 497]
    - A directory of images
    - Optional image transforms
    """

    def __init__(self, dataframe, image_dir, transform=None):
        """
        Args:
            csv_path (str): Path to the CSV file containing image names and labels
            image_dir (str): Directory containing training images
            transform (callable, optional): A torchvision transform to apply to each image
        """
        self.df = dataframe
        self.image_dir = image_dir
        self.transform = transform

        # Image filenames (e.g., 12345.jpg)
        self.image_names = self.df['Filename'].values

        # Drop 'Filename' column and convert remaining 0/1 label matrix to float32
        self.labels = self.df.drop(columns=['Filename']).values.astype(np.float32)

    def __len__(self):
        """Return total number of samples"""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            image (Tensor): Transformed image tensor
            label (Tensor): One-hot encoded label vector (shape: [498])
        """
        image_name = self.image_names[idx]
        label = torch.tensor(self.labels[idx])

        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')  # Convert to RGB if not already

        if self.transform:
            image = self.transform(image)

        return image, label


class FoodTestDataset(Dataset):
    """
    PyTorch Dataset class for loading test images (without labels).

    Returns image and filename (for submission).
    """

    def __init__(self, image_filenames, image_dir, transform=None):
        """
        Args:
            image_filenames (List[str]): List of test image filenames (e.g., from os.listdir)
            image_dir (str): Directory where test images are stored
            transform (callable, optional): A torchvision transform to apply
        """
        self.filenames = image_filenames
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        """Return total number of test images"""
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Returns:
            image (Tensor): Transformed test image tensor
            filename (str): Original image filename (used for submission)
        """
        filename = self.filenames[idx]
        image_path = os.path.join(self.image_dir, filename)

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, filename