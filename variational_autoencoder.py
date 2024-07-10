import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers, models, metrics, losses


class Sampler(layers.Layer):
    """
    Sampling layer to sample from a normal distribution with 
    mean 'emb_mean' and log variance 'emb_log_var'.
    """

    def __init__(self, **kwargs):
        super(Sampler, self).__init__(**kwargs)

    def call(self, inputs):
        emb_mean, emb_log_var = inputs
        batch_size = tf.shape(emb_mean)[0]
        dim_size = tf.shape(emb_mean)[1]

        # Use reparameterization trick to sample from the distribution
        noise = K.random_normal(shape=(batch_size, dim_size))
        return emb_mean + tf.exp(0.5 * emb_log_var) * noise


class Encoder(models.Model):
    
    def __init__(self, embedding_size=200, num_channels=128, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        
        # Embedding size
        self.embedding_size = embedding_size

        # Convolutional layers for dimensionality reduction and feature extraction
        self.conv1 = layers.Conv2D(num_channels, (3,3), strides=2, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(num_channels, (3,3), strides=2, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(num_channels, (3,3), strides=2, padding='same')
        self.bn3 = layers.BatchNormalization()
        self.conv4 = layers.Conv2D(num_channels, (3,3), strides=2, padding='same')
        self.bn4 = layers.BatchNormalization()

        self.activation = layers.LeakyReLU()
        
        # Layers to calculate mean and log of variance for each input
        self.flattening = layers.Flatten()
        self.dense_mean = layers.Dense(self.embedding_size, activation='relu', name="emb_mean")
        self.dense_log_var = layers.Dense(self.embedding_size, activation='relu', name="emb_log_var")
        
        # Sampling layer for drawing sample calculated normal distribution
        self.sampler = Sampler()

    def call(self, inputs):
        """One forward pass for given inputs"""

        # Apply convolutional layers
        x = self.conv1(inputs)
        x = self.activation(self.bn1(x))
        x = self.conv2(x)
        x = self.activation(self.bn2(x))
        x = self.conv3(x)
        x = self.activation(self.bn3(x))
        x = self.conv4(x)
        x = self.activation(self.bn4(x))

        # for use in decoder
        self.shape_before_flattening = K.int_shape(x)[1:]

        # Flatten the output from the convolutional layers
        x = self.flattening(x)
        
        # Calculate the mean and log variance
        emb_mean = self.dense_mean(x)
        emb_log_var = self.dense_log_var(x)
        
        # Draw samples from the distribution
        emb_sampled = self.sampler([emb_mean, emb_log_var])

        return emb_mean, emb_log_var, emb_sampled


class Decoder(models.Model):
    
    def __init__(self, shape_before_flattening, num_channels=128, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.shape_before_flattening = shape_before_flattening
        self.num_channels = num_channels

        # Dense layer to convert the embedding to the size of the feature vector
        # after flattening in the encoder
        self.dense1 = layers.Dense(np.prod(self.shape_before_flattening))
        self.bn_dense = layers.BatchNormalization()
        self.reshape = layers.Reshape(self.shape_before_flattening)

        # A series of transpose convolution to increase dimensionality
        self.convtr1 = layers.Conv2DTranspose(self.num_channels, kernel_size=3, strides=2, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.convtr2 = layers.Conv2DTranspose(self.num_channels, kernel_size=3, strides=2, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.convtr3 = layers.Conv2DTranspose(self.num_channels, kernel_size=3, strides=2, padding='same')
        self.bn3 = layers.BatchNormalization()
        self.convtr4 = layers.Conv2DTranspose(self.num_channels, kernel_size=3, strides=2, padding='same')
        self.bn4 = layers.BatchNormalization()

        # Reduce number of channels to input image channels
        self.conv1 = layers.Conv2D(3, (3,3), strides=1, activation='sigmoid', padding='same')

        self.activation = layers.LeakyReLU()

    def call(self, inputs):
        """One forward pass for given inputs"""

        x = self.dense1(inputs)
        x = self.activation(self.bn_dense(x))
        
        x = self.reshape(x)

        x = self.convtr1(x)
        x = self.activation(self.bn1(x))
        x = self.convtr2(x)
        x = self.activation(self.bn2(x))
        x = self.convtr3(x)
        x = self.activation(self.bn3(x))
        x = self.convtr4(x)
        x = self.activation(self.bn4(x))

        output = self.conv1(x)

        return output


class VAE(models.Model):
    
    def __init__(self, input_img_size=64, embedding_size=200, num_channels=128, beta=2000, **kwargs):
        super(VAE, self).__init__(**kwargs)
        
        # Number of channels of conv and transpose conv inside decoder and encoder
        self.num_channels = num_channels
        # Size of embedding at bottle neck of Variational Autoencoder
        self.embedding_size = embedding_size
        # weight of reconstruction loss in comparosion of KL loss
        self.beta = beta
        # Input image shape
        self.input_img_size = input_img_size

        # Create encoder
        self.enc = Encoder(embedding_size=self.embedding_size, num_channels=self.num_channels)

        # Feed a random value to calculate shape of features before flattening
        random_input = np.random.random((1, self.input_img_size, self.input_img_size, 3)).astype(np.float32)
        _, _, emb_sampled = self.enc(random_input) 

        # Create decoder
        self.dec = Decoder(shape_before_flattening=self.enc.shape_before_flattening, num_channels=self.num_channels)
        _ = self.dec(emb_sampled)

        # MSE Loss functions
        self.mse = losses.MeanSquaredError()
        # KL Divergence Loss
        self.kl = lambda emb_mean_, emb_log_var_: tf.reduce_mean(
                                                    tf.reduce_sum(
                                                        -0.5 * (1 + emb_log_var_ - tf.square(emb_mean_) - tf.exp(emb_log_var_)), 
                                                    axis=1))


        # Mean calculator for different losses during training
        self.tracker_total_loss = metrics.Mean(name="total_loss")
        self.tracker_reconstruct_loss = metrics.Mean(name="reconst_loss")
        self.tracker_kl_loss = metrics.Mean(name="kl_loss")

    def call(self, inputs):
        """One forward pass for given inputs"""
        
        # Feed input to encoder
        emb_mean, emb_log_var, emb_sampled = self.enc(inputs)

        # Reconstruct with decoder
        reconst = self.dec(emb_sampled)

        return emb_mean, emb_log_var, reconst

    @property
    def metrics(self):
        return [
            self.tracker_total_loss,
            self.tracker_reconstruct_loss,
            self.tracker_kl_loss]
    
    def train_step(self, data):
        """Perform one step traning"""
        
        with tf.GradientTape() as tape:

            # Forward pass
            emb_mean, emb_log_var, reconst = self(data, training=True)

            # Calculate reconstruction loss between input and output of VAE
            loss_recost = self.beta * self.mse(data, reconst)

            # Calculate KL divergence of predicted normal distribution for embedding and 
            # a standard normal distribution 
            loss_kl = self.kl(emb_mean, emb_log_var)

            # Total loss
            loss_total = loss_recost + loss_kl

        # calculate gradient of loss w.r.t to weights
        gradients = tape.gradient(loss_total, self.trainable_weights)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights)) 

        # Update mean of losses
        self.tracker_total_loss.update_state(loss_total)
        self.tracker_reconstruct_loss.update_state(loss_recost)
        self.tracker_kl_loss.update_state(loss_kl)

        return {
                "loss": self.tracker_total_loss.result(),
                "reconstruction_loss": self.tracker_reconstruct_loss.result(),
                "kl_loss": self.tracker_kl_loss.result()}


    def test_step(self, data):
        """Perform one step validation/test"""

        if isinstance(data, tuple):
            data = data[0]
            
        # Forward pass
        emb_mean, emb_log_var, reconst = self(data, training=False)
        
        # Calculate reconstruction loss between input and output of VAE
        loss_recost = self.beta * self.mse(data, reconst)

        # Calculate KL divergence of predicted normal distribution for embedding and 
        # a standard normal distribution 
        loss_kl = self.kl(emb_mean, emb_log_var)

        # Total loss
        loss_total = loss_recost + loss_kl

        return {
                "loss": loss_total,
                "reconstruction_loss": loss_recost,
                "kl_loss": loss_kl}
  

if __name__ == '__main__':
    
    import numpy as np

    vae_model = VAE(input_img_size=64, embedding_size=200, num_channels=128, beta=2000)

    random_input = np.random.random((2, 64, 64, 3)).astype(np.float32)
    
    out = vae_model(random_input)

    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)