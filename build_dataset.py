import os
import cv2
import numpy as np
from PIL import Image
import albumentations as A

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

# Constants
NUM_BACKGROUND_IMAGES = 6
RANDOM_SEED = 1

# Utility Functions
def get_random_size(doc_height, doc_width, factor_range):

    """
    Generate a random size based on the given factor range.
    """
    size_factor = np.random.uniform(factor_range[0], factor_range[1])
    new_h, new_w = int(size_factor * doc_height), int(size_factor * doc_width)
    return new_h, new_w


def get_random_crop(image, crop_height, crop_width):

    """
    Get random crop coordinates for an image.
    """
    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    return y, x, y + crop_height, x + crop_width


def apply_transformations(image, mask, transformer, distortion_scale):

    """
    Apply perspective transformations to an image and its mask.
    """
    W, H = image.size
    torch.manual_seed(RANDOM_SEED)
    startpoints, endpoints = transformer.get_params(W, H, distortion_scale=distortion_scale)

    transformed_image = F.perspective(image, startpoints, endpoints, fill=0, interpolation=T.InterpolationMode.NEAREST)
    transformed_mask = F.perspective(mask, startpoints, endpoints, fill=0, interpolation=T.InterpolationMode.NEAREST)

    return transformed_image, transformed_mask


def setup_augmentations():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=45, p=0.5),
        # Add more transformations as needed
    ])


def merge_images(background, transformed_image, mask):

    """
    Merge the transformed image onto the background using the provided mask.
    """
    # Convert PIL Images to NumPy arrays
    transformed_image_np = np.array(transformed_image)
    mask_np = np.array(mask)

    # Ensure the mask is boolean
    mask_np = mask_np > 128  # Assuming the mask is a binary image

    # Prepare an empty canvas with the size of the background
    merged_image = np.zeros_like(background)

    # Place transformed image onto the background
    for c in range(3):  # Loop over color channels
        merged_image[:, :, c] = np.where(mask_np, transformed_image_np[:, :, c], background[:, :, c])

    return merged_image


def save_image_and_mask(image, mask, index, img_dir, mask_dir):
    """
    Save the image and mask to the specified directories.
    """
    # Create filenames
    img_filename = os.path.join(img_dir, f"processed_img_{index:04d}.png")
    mask_filename = os.path.join(mask_dir, f"processed_mask_{index:04d}.png")

    # Save image and mask
    cv2.imwrite(img_filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV
    cv2.imwrite(mask_filename, mask)

    print(f"Saved Image: {img_filename}, Mask: {mask_filename}")


def augment_image_and_mask(image, mask, augmentation_pipeline):
    """
    Apply a series of augmentations to an image and its mask.
    """
    augmented = augmentation_pipeline(image=image, mask=mask)
    return augmented["image"], augmented["mask"]


def process_image(params):
    """
    Main processing function for each image. This function handles the entire
    image processing pipeline.
    """
    # Extracting parameters
    prc_id, doc_images, doc_masks, background_images, gen_img_dir, gen_msk_dir, start_idx = (
        params["id"], params["DOC_IMGS"], params["DOC_MSKS"], params["BCK_IMGS"],
        params["GEN_IMG_DIR"], params["GEN_MSK_DIR"], params["start_idx"]
    )

    print(f"[INFO] Starting process {prc_id}")

    # Image augmentations setup
    augmentation_pipeline = setup_augmentations()

    # Transformer setup for perspective transformations
    distortion_scale = 0.55
    perspective_transformer = T.RandomPerspective(distortion_scale=distortion_scale, p=0.7, interpolation=T.InterpolationMode.NEAREST)

    # Process each document image
    for doc_index, (img_path, msk_path) in enumerate(zip(doc_images, doc_masks), start_idx):
        original_image = Image.open(img_path).convert("RGB")
        original_mask = Image.open(msk_path).convert("L")  # Assuming mask is grayscale

        # Generate transformed images and masks
        transformed_image, transformed_mask = apply_transformations(original_image, original_mask, perspective_transformer, distortion_scale)

        # Select a random background image
        random_background_index = np.random.choice(range(len(background_images)))
        background_image_path = background_images[random_background_index]
        background_image = cv2.imread(background_image_path, cv2.IMREAD_COLOR)[:, :, ::-1]  # Convert BGR to RGB

        # Resize and place transformed image on background
        new_height, new_width = get_random_size(transformed_image.height, transformed_image.width, factor_range=(1.1, 1.4))
        background_image = cv2.resize(background_image, (new_width, new_height), cv2.INTER_CUBIC)
        ymin, xmin, ymax, xmax = get_random_crop(background_image, transformed_image.height, transformed_image.width)
        cropped_background = background_image[ymin:ymax, xmin:xmax, :] / 255.0

        # Merge transformed image and cropped background
        merged_image = merge_images(cropped_background, transformed_image, transformed_mask)

        # Apply augmentations
        augmented_image, augmented_mask = augment_image_and_mask(merged_image, transformed_mask, augmentation_pipeline)

        # Save the processed image and mask
        save_image_and_mask(augmented_image, augmented_mask, doc_index, gen_img_dir, gen_msk_dir)

    print(f"[INFO] Finishing process")
