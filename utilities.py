import numpy as np
import tensorflow as tf

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