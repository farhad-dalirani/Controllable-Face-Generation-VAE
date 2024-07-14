import streamlit as st
import os
import json
import numpy as np
import tensorflow as tf
from variational_autoencoder import VAE
from utilities import get_split_data, get_random_images
from synthesis import generate, reconstruct

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
    to reconstruct, then concatenate images.

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

def latent_arithmetic(config, model_vae, attribute_name, num_images=10):
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
    latent_attribute_vector = np.array(st.session_state["attribute_vectors"][attribute_name])
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
        st.markdown("Perform arithmetic operations in the latent space of faces based on selected attributes")

        # Dropdown to select attribute keys
        st.session_state.attribute_key = st.selectbox("Select Attribute Key", list(st.session_state["attribute_vectors"].keys()),
                                                       index=list(st.session_state["attribute_vectors"].keys()).index('Blond_Hair'))

        if 'images_latent_arith' not in st.session_state:
            st.session_state.images_latent_arith = latent_arithmetic(config, model_vae, attribute_name=st.session_state.attribute_key, num_images=10)

        if st.button('Perform Latent Space Arithmetic'):
            st.session_state.images_latent_arith = latent_arithmetic(config, model_vae, attribute_name=st.session_state.attribute_key, num_images=7)

        st.markdown("""
        <div style="background-color:#f0f0f0;padding:10px;border-radius:5px;">
            <p style="text-align:center;font-size:18px;font-weight:bold;">← Subtract - [Latent Vector of the Selected Attribute] - Add →</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div style="text-align:center;">', unsafe_allow_html=True)
        st.image(st.session_state.images_latent_arith)
        st.markdown('</div>', unsafe_allow_html=True)

    elif app_feature == 'Morph Faces':
        pass
        

if __name__ == '__main__':
    
    # Define the path to the configuration file
    config_path = 'config/config.json'

    # Open and read the configuration file to load settings
    with open(config_path, 'r') as file:
        config = json.load(file)

    # Check if the VAE model is already loaded in the session state
    if "model" not in st.session_state.keys():
        # Instantiate and load the VAE model with specified parameters
        st.session_state["model"] = VAE(
            input_img_size=config["input_img_size"], 
            embedding_size=config["embedding_size"], 
            num_channels=config["num_channels"], 
            beta=config["beta"])
        # Load model weights from the specified path
        st.session_state["model"].load_weights(os.path.join(config["model_save_path"], "vae.keras"))
        # Print model summary for verification
        st.session_state["model"].summary()

    # Check if the validation dataset is already loaded in the session state
    if "val_data" not in st.session_state.keys():
        # Load and split the dataset, storing validation data in session state
        train, st.session_state["val_data"] = get_split_data(config=config)
    
    # Check if attribute vectors for latent space are already loaded in the session state
    if "attribute_vectors" not in st.session_state.keys():
        # Open and read attribute vectors from the specified file
        with open("attributes_embedings/attributes_embedings.json", 'r') as f:
            st.session_state["attribute_vectors"] = json.load(f)

    # Run the main application function, handling SystemExit to allow for graceful exit    
    try:
        main(config, st.session_state["model"], st.session_state["val_data"])
    except SystemExit:
        pass
