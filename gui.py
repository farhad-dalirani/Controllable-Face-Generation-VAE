import streamlit as st
import os
import json
import numpy as np
import tensorflow as tf
from variational_autoencoder import VAE
from utilities import get_split_data, get_random_images
from synthesis import generate, reconstruct

def generate_images(config, model_vae):
    """
    Generate new images with VAE and concatenate images to create one image
    """
    images_list = generate(decoder=model_vae.dec, emd_size=config["embedding_size"],
                            num_generated_imgs=70)
    rows = []
    for i in range(7):
        rows.append(np.concatenate((images_list[(i*10):((i+1)*10)]), axis=1))
    all_images = np.concatenate(rows, axis=0)
    return all_images

def reconstruct_images(model_vae, validation):
    """
    Randomly select some faces from CelebA dataset, feed to VAE to 
    reconstruct, then concatenate images
    """
    # Get some random images from dataset
    images = get_random_images(dataset=validation, num_images=40)
    
    # Reconstruct some images by VAE
    list_imgs_recons = reconstruct(vae=model_vae, input_images=images)
    
    rows = []
    for i in range(4):
        row_rec = np.concatenate((list_imgs_recons[(i*10):((i+1)*10)]), axis=1)
        row_org = np.concatenate(([images[i] for i in range((i*10), ((i+1)*10))]), axis=1)
        rows.append(np.concatenate((row_org, row_rec), axis=0))
    all_images = np.concatenate(rows, axis=0)
    return all_images

def main(config, model_vae, validation):
    st.title("Controllable Face Generation VAE")
    st.sidebar.title("Settings")
    st.sidebar.subheader("Features")

    # Select between different features
    app_feature = st.sidebar.selectbox("Choose the App Mode", [
                                        'Generate Faces', 
                                        'Reconstruct Faces', 
                                        'Face Latent Space Arithmetic', 
                                        'Morph Faces'])

    if app_feature == 'Generate Faces':
        st.markdown("Randomly select vectors from a standard normal distribution and feed them to the decoder to generate new faces")

        if 'images_gen' not in st.session_state:
            st.session_state.images_gen = generate_images(config, model_vae)

        if st.button('Generate New Faces'):
            st.session_state.images_gen = generate_images(config, model_vae)

        st.image(st.session_state.images_gen)

    elif app_feature == 'Reconstruct Faces':
        st.markdown("Randomly select faces from the CelebA dataset, feed them to a variational autoencoder, and depict the reconstructed faces")
        
        if 'images_rec' not in st.session_state:
            st.session_state.images_rec = reconstruct_images(model_vae, validation)

        if st.button('Reconstruct New Faces'):
            st.session_state.images_rec = reconstruct_images(model_vae, validation)

        st.image(st.session_state.images_rec)

    elif app_feature == 'Face Latent Space Arithmetic':
        pass
    elif app_feature == 'Morph Faces':
        pass

if __name__ == '__main__':
    # Path to config file
    config_path = 'config/config.json'

    # Open and read the config file
    with open(config_path, 'r') as file:
        config = json.load(file)

    if "model" not in st.session_state.keys():
        # Load VAE model
        st.session_state["model"] = VAE(
            input_img_size=config["input_img_size"], 
            embedding_size=config["embedding_size"], 
            num_channels=config["num_channels"], 
            beta=config["beta"])
        st.session_state["model"].load_weights(os.path.join(config["model_save_path"], "vae.keras"))
        st.session_state["model"].summary()

    # Load dataset
    if "val_data" not in st.session_state.keys():
        train, st.session_state["val_data"] = get_split_data(config=config)
    

    try:
        main(config, st.session_state["model"], st.session_state["val_data"])
    except SystemExit:
        pass
