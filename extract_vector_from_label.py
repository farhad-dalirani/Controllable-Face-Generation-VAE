import os
import json
import numpy as np
import pandas as pd 
from tensorflow.keras import utils
from variational_autoencoder import VAE
import tensorflow as tf

def extract_attribute_vector_from_label(vae, embedding_size, train_data, df_attributes):
    """
        For each attribute finds a vector in latent space that adding it to
        a latent vector make the generated image has that feature
    """
    
    # Get all attribute names
    attributes_names = df_attributes.columns[1:]
    
    # Average embeding vector for sample with i-th attribute is positive  
    attributes_emb_sum_pos = {i:np.zeros(shape=embedding_size, dtype="float32") for i in range(len(attributes_names))}
    attributes_emb_num_pos = {i:0 for i in range(len(attributes_names))}
    # Average embeding vector for sample with i-th attribute is negative  
    attributes_emb_sum_neg = {i:np.zeros(shape=embedding_size, dtype="float32") for i in range(len(attributes_names))}
    attributes_emb_num_neg = {i:0 for i in range(len(attributes_names))}

    # For each batch in train set
    for batch_i in train_data.as_numpy_iterator():
        
        # Images in a batch
        images = batch_i[0]
        images = tf.cast(images, "float32") / 255.0
        
        # Image file name of images in a batch
        images_file_numbers = batch_i[1]
        
        # Feed images in the batch to Variational Autoencoder
        batch_emb, _, _ = vae.predict(images)
        
        # Update sum of embedings for each attribute
        for idx, emb_i in enumerate(batch_emb):
            
            # Ground truth attributes associate with image
            df_row = images_file_numbers[idx] - 1
            attributes_img_i = df_attributes.loc[df_row].to_list()[1:]

            # For each attribute update some of embedings
            for attribute_j_idx, attribute_j_val in enumerate(attributes_img_i):
                
                if attribute_j_val == -1:    
                    attributes_emb_sum_neg[attribute_j_idx] += emb_i
                    attributes_emb_num_neg[attribute_j_idx] += 1
                elif attribute_j_val == 1:
                    attributes_emb_sum_pos[attribute_j_idx] += emb_i
                    attributes_emb_num_pos[attribute_j_idx] += 1
                else:
                    raise ValueError('GT Attribute Value: {} is wrong'.format(attribute_j_val))             

    
    # Calculate embeding mean of positive and negative samples for each attribute
    attributes_emb_mean_pos = {}
    attributes_emb_mean_neg = {}
    for key_att_i in attributes_emb_sum_pos.keys():
        attributes_emb_mean_pos[key_att_i] = attributes_emb_sum_pos[key_att_i] / attributes_emb_num_pos[key_att_i] 
        attributes_emb_mean_neg[key_att_i] = attributes_emb_sum_neg[key_att_i] / attributes_emb_num_neg[key_att_i] 
    
    # Calculate Vector for each feature that adding it to embeding causes appearance of that feature inside 
    # generated image
    attributes_vectors = {}
    for key_att_i in attributes_emb_sum_pos.keys():
        attributes_vectors[attributes_names[key_att_i]] = attributes_emb_mean_pos[key_att_i] - attributes_emb_mean_neg[key_att_i]

    return attributes_vectors



if __name__ == '__main__':
    
    # Path to config file
    config_path = 'config/config.json'

    # Open and read the config file
    with open(config_path, 'r') as file:
        config = json.load(file)

    # Read classes csv file, faces attributes
    df_attributes = pd.read_csv(filepath_or_buffer=os.path.join(config["dataset_dir"], 'list_attr_celeba.csv'))
    
    print(df_attributes.head())

    # Create training dataset (number of file name as class)
    validation_split = 0.2
    train_data = utils.image_dataset_from_directory(
        os.path.join(config["dataset_dir"], "img_align_celeba"),
        labels=[i+1 for i in range(len(df_attributes))],
        color_mode="rgb",
        image_size=(config["input_img_size"], config["input_img_size"]),
        batch_size=config["batch_size"],
        shuffle=True,
        seed=0,
        validation_split=validation_split,
        subset="training",
        interpolation="bilinear",
    )

    # Load VEA model
    model_vae = VAE(
        input_img_size=config["input_img_size"], 
        embedding_size=config["embedding_size"], 
        num_channels=config["num_channels"], 
        beta=config["beta"])
    model_vae.load_weights(os.path.join(config["model_save_path"],  "vae.keras"))
    model_vae.summary()

    # Caluculate each attribute vector in latent space
    attributes_vectors = extract_attribute_vector_from_label(
        vae=model_vae, embedding_size=config["embedding_size"], 
        train_data=train_data, df_attributes=df_attributes)

    print(attributes_vectors)
     
    # Function to convert ndarray to list
    def convert_ndarray_to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert ndarray to list
        raise TypeError(f'Object of type {type(obj)} is not JSON serializable')

    # Save to a JSON file
    with open('attributes_embedings/attributes_embedings.json', 'w') as json_file:
        json.dump(attributes_vectors, json_file, default=convert_ndarray_to_list)

    