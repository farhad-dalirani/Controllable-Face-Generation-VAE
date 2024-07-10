import tensorflow as tf

def data_preprocess(image):
    """Convert UINT images to float in range 0, 1"""
    image = tf.cast(image, "float32") / 255.0
    return image
