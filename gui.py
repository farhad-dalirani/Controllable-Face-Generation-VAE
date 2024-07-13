import streamlit as st
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import utils
from variational_autoencoder import VAE
from utilities import get_split_data, get_random_images
from synthesis import generate

def generate_images(config, model_vae):
    images_list = generate(decoder=model_vae.dec, emd_size=config["embedding_size"], num_generated_imgs=70)
    rows = []
    for i in range(7):
        rows.append(np.concatenate((images_list[(i*10):((i+1)*10)]), axis=1))
    all_images = np.concatenate(rows, axis=0)
    return all_images

def main(config):
    st.title("Controllable Face Generation VAE")
    st.sidebar.title("Settings")
    st.sidebar.subheader("Features")

    # Select between different features
    app_feature = st.sidebar.selectbox("Choose the App Mode", ['Generate Faces', 'Face Latent Space Arithmetic', 'Morph Faces'])

    if app_feature == 'Generate Faces':
        st.markdown("Randomly select 10 vectors from a standard normal distribution and feed them to the decoder to generate new faces")

        if 'images' not in st.session_state:
            st.session_state.images = generate_images(config, model_vae)

        if st.button('Generate New Faces'):
            st.session_state.images = generate_images(config, model_vae)

        st.image(st.session_state.images)

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

    # Load VEA model
    model_vae = VAE(
        input_img_size=config["input_img_size"], 
        embedding_size=config["embedding_size"], 
        num_channels=config["num_channels"], 
        beta=config["beta"])
    model_vae.load_weights(os.path.join(config["model_save_path"], "vae.keras"))
    model_vae.summary()

    try:
        main(config)
    except SystemExit:
        pass
