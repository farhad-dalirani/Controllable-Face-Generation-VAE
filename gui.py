import streamlit as st
import os
import json
import numpy as np
from variational_autoencoder import VAE
from utilities import get_split_data
from synthesis import generate_images, reconstruct_images, latent_arithmetic_on_images, morph_images, generate_images_with_selected_attributes_vectors


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
        st.markdown("Randomly select vectors from a standard normal distribution. If any attributes are selected, add the selected attributes' latent space vector to the samples and feed them to the decoder to generate new faces.")

        # List of options for checkboxes
        options = st.session_state["attribute_vectors"].keys()

        # Initialize session state if not already done
        if 'selected_options' not in st.session_state:
            st.session_state['selected_options'] = {option: False for option in options}
        if 'attribute_sliders' not in st.session_state:
            st.session_state['attribute_sliders'] = {option: 1.0 for option in options}

        # Create checkboxes and sliders, and update session state
        st.sidebar.markdown("### Select Attributes")
        for option in options:
            st.session_state['selected_options'][option] = st.sidebar.checkbox(option, value=st.session_state['selected_options'][option])
            if st.session_state['selected_options'][option]:
                st.session_state['attribute_sliders'][option] = st.sidebar.slider(f"{option} value", -3.0, 3.0, value=st.session_state['attribute_sliders'][option])

        # Generate images with selected options
        selected_options = [option for option, selected in st.session_state['selected_options'].items() if selected]
        attributes_vectors = []
        for attribute_key in selected_options:
            attributes_vectors.append(np.array(st.session_state["attribute_vectors"][attribute_key]) * st.session_state['attribute_sliders'][attribute_key])
        
        if 'images_generated' not in st.session_state:
            st.session_state.images_generated = generate_images_with_selected_attributes_vectors(decoder=model_vae.dec, emd_size=config['embedding_size'], attributes_vectors=attributes_vectors)
        if st.button('Generate New Faces'):
            st.session_state.images_generated = generate_images_with_selected_attributes_vectors(decoder=model_vae.dec, emd_size=config['embedding_size'], attributes_vectors=attributes_vectors)

        st.image(st.session_state.images_generated)
        
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
            st.session_state.images_latent_arith = latent_arithmetic_on_images(
                                                        config, model_vae,
                                                        attribute_vector=np.array(st.session_state["attribute_vectors"][st.session_state.attribute_key]),
                                                        num_images=10)

        if st.button('Perform Latent Space Arithmetic'):
            st.session_state.images_latent_arith = latent_arithmetic_on_images(
                                                        config, 
                                                        model_vae, 
                                                        attribute_vector=np.array(st.session_state["attribute_vectors"][st.session_state.attribute_key]), 
                                                        num_images=7)

        st.markdown("""
        <div style="background-color:#f0f0f0;padding:10px;border-radius:5px;">
            <p style="text-align:center;font-size:18px;font-weight:bold;">← Subtract | Latent Vector of {} | Add →</p>
        </div>
        """.format(st.session_state.attribute_key), unsafe_allow_html=True)
        
        st.markdown('<div style="display: flex; justify-content: center; align-items: center;">', unsafe_allow_html=True)
        st.image(st.session_state.images_latent_arith)
        st.markdown('</div>', unsafe_allow_html=True)

    elif app_feature == 'Morph Faces':
        st.markdown("Generate faces and blend them together by calculating points between the embeddings of two faces.")

        if 'images_morph' not in st.session_state:
            st.session_state.images_morph = morph_images(config, model_vae, num_images=10)

        if st.button('Morph Faces'):
            st.session_state.images_morph = morph_images(config, model_vae, num_images=10)
        
        st.markdown("""
        <div style="background-color:#f0f0f0;padding:10px;border-radius:5px;">
            <p style="text-align:center;font-size:18px;font-weight:bold;">→ Blend Pairs of Faces ←</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.image(st.session_state.images_morph)

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
