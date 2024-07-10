import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import utils
from variational_autoencoder import VAE
from utilities import data_preprocess, get_random_images

def generate(decoder, emd_size=200, num_generated_imgs=10):
    """
        Generate new images by drawing samples from 
        standard normal distribution and feeding them into decoder
    """
    # Draw sample from standard normal distribution
    mean = np.zeros(emd_size)
    cov = np.eye(emd_size)
    samples = np.random.multivariate_normal(mean, cov, size=num_generated_imgs)

    # Feed embedings to decoder
    outputs = decoder.predict(samples)

    images_list = [outputs[i] for i in range(outputs.shape[0])]

    return images_list

def reconstruct(vae, input_images):
    """Feed images to VAE and get reconstruction images"""
    _, _, reconst = vae.predict(input_images)

    images_list = [reconst[i] for i in range(reconst.shape[0])]

    return images_list

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt

    # Path to config file
    config_path = 'config/config.json'

    # Open and read the config file
    with open(config_path, 'r') as file:
        config = json.load(file)
    
    # Load VEA model
    model_vae = VAE(
        input_img_size=config["input_img_size"], 
        embedding_size=config["embedding_size"], 
        num_channels=config["num_channels"], 
        beta=config["beta"])
    model_vae.load_weights(os.path.join(config["model_save_path"], "vae.keras"))
    model_vae.summary()

    images_list = generate(decoder=model_vae.dec, emd_size=config["embedding_size"], num_generated_imgs=10)

    for img_i in images_list:
        plt.figure()
        plt.imshow(img_i)
    plt.show()

    # Load dataset
    train_data = utils.image_dataset_from_directory(
        os.path.join(config["dataset_dir"], "img_align_celeba"),
        labels=None,
        color_mode="rgb",
        image_size=(config["input_img_size"], config["input_img_size"]),
        batch_size=config["batch_size"],
        shuffle=True,
        seed=0,
        interpolation="bilinear",
    )
    # UINT images to float in range 0, 1
    train = train_data.map(lambda x: data_preprocess(x))

    # Get some random images from dataset
    images = get_random_images(dataset=train, num_images=10) 

    # Reconstruct some images by VAE
    list_imgs_recons = reconstruct(vae=model_vae, input_images=images)
    
    for idx, img_i in enumerate(list_imgs_recons):
        plt.figure()
        plt.imshow(np.hstack((images[idx], img_i)))
    plt.show()
    