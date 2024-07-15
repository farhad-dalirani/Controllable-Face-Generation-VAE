import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import utils
from variational_autoencoder import VAE
from utilities import get_split_data, get_random_images

def generate(decoder, emd_size=200, num_generated_imgs=10, return_samples=False):
    """
    Generate new images by drawing samples from a standard normal distribution 
    and feeding them into the decoder of VAE.

    Parameters:
    - decoder: The decoder model of VAE to generate images from embeddings.
    - emd_size: The size of the embedding vector at end of encoder (default is 200).
    - num_generated_imgs: The number of images to generate (default is 10).
    - return_samples: Whether to return the samples along with the images (default is False).

    Returns:
    - images_list: A list of generated images.
    - samples (optional): The samples drawn from the standard normal distribution, 
      returned if return_samples is True.
    """

    # Draw samples from a standard normal distribution
    mean = np.zeros(emd_size)
    cov = np.eye(emd_size)
    samples = np.random.multivariate_normal(mean, cov, size=num_generated_imgs)

    # Feed the embeddings to the decoder to generate images
    outputs = decoder.predict(samples)

    # Create a list of generated images
    images_list = [outputs[i] for i in range(outputs.shape[0])]

    # Return the generated images, and optionally the samples
    if return_samples == False:
        return images_list
    else:
        return images_list, samples

def reconstruct(vae, input_images):
    """
    Feed images to a Variational Autoencoder (VAE) and get reconstructed images.

    Parameters:
    - vae: The Variational Autoencoder model used for reconstruction.
    - input_images: The images to be fed into the VAE for reconstruction.

    Returns:
    - images_list: A list of reconstructed images.
    """
    
    # Feed the input images to the VAE and get the reconstructed images
    _, _, reconst = vae.predict(input_images)

    # Create a list of reconstructed images
    images_list = [reconst[i] for i in range(reconst.shape[0])]

    return images_list


def generate_images(config, model_vae, num_images=70):
    """
    Generate new images with VAE and concatenate images to create one image.

    Args:
        config (dict): Configuration dictionary containing embedding size.
        model_vae (object): Trained VAE model with decoder attribute.
        num_images (int, optional): Number of images to generate. Defaults to 70.

    Returns:
        np.ndarray: Concatenated image of all generated images.
    """
    
    # Generate a list of images using the VAE decoder
    images_list = generate(decoder=model_vae.dec, 
                           emd_size=config["embedding_size"],
                           num_generated_imgs=num_images)
    
    rows = []
    # Concatenate images into rows of 10 images each
    for i in range(7):
        rows.append(np.concatenate((images_list[(i*10):((i+1)*10)]), axis=1))
    
    # Concatenate all rows into a single image
    all_images = np.concatenate(rows, axis=0)
    
    return all_images


def reconstruct_images(model_vae, validation):
    """
    Randomly select some faces from the CelebA dataset, feed them to the VAE 
    to reconstruct, then concatenate images to one image.

    Args:
        model_vae (object): Trained VAE model.
        validation (Dataset): Validation dataset containing CelebA images.

    Returns:
        np.ndarray: Concatenated image of original and reconstructed images.
    """

    # Get some random images from the validation dataset
    images = get_random_images(dataset=validation, num_images=40)
    
    # Reconstruct the selected images using the VAE
    list_imgs_recons = reconstruct(vae=model_vae, input_images=images)
    
    rows = []
    # Concatenate original and reconstructed images in rows of 10
    for i in range(4):
         # Concatenate reconstructed images for the current row
        row_rec = np.concatenate((list_imgs_recons[(i*10):((i+1)*10)]), axis=1)
        # Concatenate original images for the current row
        row_org = np.concatenate(([images[i] for i in range((i*10), ((i+1)*10))]), axis=1)
        # Concatenate the original and reconstructed rows vertically
        rows.append(np.concatenate((row_org, row_rec), axis=0))
    
    # Concatenate all rows into a single image
    all_images = np.concatenate(rows, axis=0)
    
    return all_images


def latent_arithmetic_on_images(config, model_vae, attribute_vector, num_images=10):
    """
    Increase and decrease an attribute inside generated faces by latent space arithmetic.

    Args:
        config (dict): Configuration dictionary containing embedding size and input image size.
        model_vae (object): Trained VAE model.
        attribute_name (str): Name of the attribute to modify in the latent space.
        num_images (int, optional): Number of images to generate. Defaults to 10.

    Returns:
        np.ndarray: Concatenated image showing the effect of the attribute change across generated images.
    """
    
    # Draw samples from a standard normal distribution
    mean = np.zeros(config["embedding_size"])
    cov = np.eye(config["embedding_size"])
    sampled_embds = np.random.multivariate_normal(mean, cov, size=num_images)

    # Retrieve the latent vector for the specified attribute
    latent_attribute_vector = attribute_vector.copy()
    latent_attribute_vector = np.reshape(latent_attribute_vector, newshape=(1, -1))

    cols = []
    # Modify the latent space by adding different levels of the attribute vector
    for i in range(-3, 4):

        # Adjust the latent embeddings by adding the attribute vector scaled by i
        sampled_embds_new = sampled_embds + i * latent_attribute_vector

        # Decode the adjusted embeddings to generate images
        outputs = model_vae.dec.predict(sampled_embds_new)
        images_list = [outputs[i] for i in range(outputs.shape[0])]
        images_level_i = np.concatenate(images_list, axis=0)

        cols.append(images_level_i)

        # Add a separator between different levels of attribute change
        if (i == -1) or (i == 0):
            cols.append(np.ones(shape=(num_images * config["input_img_size"], config["input_img_size"], 3)))

    # Concatenate all columns to create the final image
    image = np.concatenate(cols, axis=1)

    return image


def morph_images(config, model_vae, num_images=10):
    """
    Morphs images by blending embeddings of two faces using a VAE model.

    Parameters:
    config (dict): Configuration dictionary containing 'embedding_size' and 'input_img_size'.
    model_vae (VAE): Pre-trained VAE model used for image generation.
    num_images (int): Number of images to generate for each blend level. Default is 10.

    Returns:
    np.ndarray: Final concatenated image showing the morphing process.
    """
    
    # Draw samples from a standard normal distribution
    mean = np.zeros(config["embedding_size"])
    cov = np.eye(config["embedding_size"])
    left_sampled_embds = np.random.multivariate_normal(mean, cov, size=num_images)
    right_sampled_embds = np.random.multivariate_normal(mean, cov, size=num_images)

    cols = []
    # Modify the latent space by adding different levels of the attribute vector
    for alpha in np.arange(0.0, 1.1, 0.1):
        
        # Adjust the latent embeddings by adding the attribute vector scaled by i
        sampled_embds_new = (1 - alpha) * left_sampled_embds + alpha * right_sampled_embds

        # Decode the embeddings to generate images
        outputs = model_vae.dec.predict(sampled_embds_new)
        images_list = [outputs[i] for i in range(outputs.shape[0])]
        images_level_i = np.concatenate(images_list, axis=0)

        cols.append(images_level_i)

        # Add a separator between different levels of attribute change
        if (alpha == 0) or (alpha == 0.9):
            cols.append(np.ones(shape=(num_images * config["input_img_size"], config["input_img_size"], 3)))

    # Concatenate all columns to create the final image
    image = np.concatenate(cols, axis=1)

    return image


def generate_images_with_selected_attributes_vectors(decoder, emd_size=200, attributes_vectors=[], num_generated_imgs=70):
    """
    Generates images with selected attribute vectors using a decoder model.

    Args:
        decoder (Model): The decoder model to generate images.
        emd_size (int): The size of the embedding vectors.
        attributes_vectors (list): List of attribute vectors to add to the sampled vectors.
        num_generated_imgs (int): The number of images to generate.

    Returns:
        np.array: A single image array containing all generated images concatenated.
    """
        
    # Draw samples from a standard normal distribution
    mean = np.zeros(emd_size)
    cov = np.eye(emd_size)
    samples = np.random.multivariate_normal(mean, cov, size=num_generated_imgs)

    # Add attribute vectors to the sampled vectors to generate new images
    for attribute_i in attributes_vectors:
        samples = samples + np.reshape(attribute_i, newshape=(1, emd_size))

    # Feed the embeddings to the decoder to generate images
    outputs = decoder.predict(samples)

    # Create a list of generated images
    images_list = [outputs[i] for i in range(outputs.shape[0])]

    rows = []
    # Concatenate images into rows of 10 images each
    for i in range(num_generated_imgs//10):
        rows.append(np.concatenate((images_list[(i*10):((i+1)*10)]), axis=1))
    
    # Concatenate all rows into a single image
    all_images = np.concatenate(rows, axis=0)
    
    return all_images
   

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
    model_vae.load_weights(os.path.join(config["model_save_path"],  "vae.keras"))
    model_vae.summary()

    images_list = generate(decoder=model_vae.dec, emd_size=config["embedding_size"], num_generated_imgs=10)

    for img_i in images_list:
        plt.figure()
        plt.imshow(img_i)
    plt.show()

    # Load dataset
    train, validation = get_split_data(config=config)

    # Get some random images from dataset
    images = get_random_images(dataset=validation, num_images=10) 

    # Reconstruct some images by VAE
    list_imgs_recons = reconstruct(vae=model_vae, input_images=images)
    
    for idx, img_i in enumerate(list_imgs_recons):
        plt.figure()
        plt.imshow(np.hstack((images[idx], img_i)))
    plt.show()
