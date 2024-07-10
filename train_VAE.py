import os
import json
import numbers as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import utils, optimizers, callbacks
from variational_autoencoder import VAE
from utilities import data_preprocess

def train_variational_autoencoder(config, train):
    """ Train Variatioanl Autoencoder """

    # Create Variatioanl Autoencoder
    model_vae = VAE(
        input_img_size=config["input_img_size"], 
        embedding_size=config["embedding_size"], 
        num_channels=config["num_channels"], 
        beta=config["beta"])

    # Optimizer
    opz = optimizers.Adam(learning_rate=config["lr"])

    # Compile model
    model_vae.compile(optimizer=opz)

    # Checkpoint and logging
    model_checkpoint_clbk = callbacks.ModelCheckpoint(
        filepath="./model_weights/checkpoint/checkpoint.keras",
        save_weights_only=False,
        save_freq="epoch",
        monitor="loss",
        mode="min",
        save_best_only=True,
        verbose=0)
    tensorboard_clbk = callbacks.TensorBoard(
                            log_dir="./model_weights/logs")

    # Train model
    model_vae.fit(
    train,
    epochs=config["max_epoch"],
    callbacks=[
        model_checkpoint_clbk,
        tensorboard_clbk])

    # Save encoder, decoder, VAE
    model_vae.save(os.path.join(config["model_save_path"], "vae.keras"))
    model_vae.enc.save(os.path.join(config["model_save_path"], "encoder.keras"))
    model_vae.dec.save(os.path.join(config["model_save_path"], "decoder.keras"))

if __name__ == '__main__':
    
    # Path to config file
    config_path = 'config/config.json'

    # Open and read the config file
    with open(config_path, 'r') as file:
        config = json.load(file)
    
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

    train_variational_autoencoder(config=config, train=train)