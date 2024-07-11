import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import utils

def data_preprocess(image):
    """Convert UINT images to float in range 0, 1"""
    image = tf.cast(image, "float32") / 255.0
    return image

def get_random_images(dataset, num_images=10):
    """Get random images from tf.data.Dataset"""
    all_images = []
    for batch in dataset:
        images = batch.numpy()  
        all_images.extend(images) 
        if len(all_images) >= num_images:
            break

    # Randomly select n images
    selected_idx = np.random.choice([i for i in range(len(all_images))], size=num_images, replace=False)
    selected_images = [all_images[idx] for idx in selected_idx]

    return np.array(selected_images)

def get_split_data(config, validation_split=0.2):
    """Return train and validation split"""

    # Create training dataset
    train_data = utils.image_dataset_from_directory(
        os.path.join(config["dataset_dir"], "img_align_celeba"),
        labels=None,
        color_mode="rgb",
        image_size=(config["input_img_size"], config["input_img_size"]),
        batch_size=config["batch_size"],
        shuffle=True,
        seed=0,
        validation_split=validation_split,
        subset="training",
        interpolation="bilinear",
    )

    # Create validation dataset
    validation_data = utils.image_dataset_from_directory(
        os.path.join(config["dataset_dir"], "img_align_celeba"),
        labels=None,
        color_mode="rgb",
        image_size=(config["input_img_size"], config["input_img_size"]),
        batch_size=config["batch_size"],
        shuffle=True,
        seed=0,
        validation_split=validation_split,
        subset="validation",
        interpolation="bilinear",
    )

    # Convert UINT images to float in range 0, 1
    train = train_data.map(lambda x: data_preprocess(x))
    validation = validation_data.map(lambda x: data_preprocess(x))

    return train, validation