import numpy as np
import tensorflow as tf
import keras
from keras import layers

from dbscan import show_cluster, dbscan_classifier, create_dbscan

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

def auto_model(features, filepath, percentile = 90):

    X_train = np.vstack([features])

    input_shape = X_train.shape[1]
    encoding_dim = 20  # Can adjust the size of the encoded representation as needed

    input = Input(shape=(input_shape,))
    encoded = Dense(64, activation='relu')(input) # First hidden layer
    encoded = Dense(32, activation='relu')(encoded)   # Second hidden layer
    encoded = Dense(encoding_dim, activation='relu')(encoded)   # Third hidden layer
    decoded = Dense(32, activation='relu')(encoded)   # First decoder layer
    decoded = Dense(64, activation='relu')(decoded)   # Second decoder layer
    decoded = Dense(input_shape, activation='sigmoid')(decoded) # Third decoder layer

    autoencoder = Model(input, decoded)

    # Compile the model
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Train the autoencoder
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, shuffle=True, validation_data=(X_train, X_train))
    autoencoder.save(filepath)

    decoded_values = autoencoder.predict(X_train)
    mse_train = np.mean(np.square(X_train - decoded_values), axis=1)

    threshold = np.percentile(mse_train, percentile)

    return threshold

@keras.saving.register_keras_serializable()
class Sampling(layers.Layer):

	def call(self, inputs):
		mean, log_var = inputs

		batch = tf.shape(mean)[0]
		dim = tf.shape(mean)[1]
		epsilon = tf.random.normal(shape=(batch, dim))
		return mean + tf.exp(0.5 * log_var) * epsilon

class VAE(keras.Model):
	def __init__(self, encoder, decoder, **kwargs):
		super().__init__(**kwargs)
		self.encoder = encoder
		self.decoder = decoder
		self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
		self.reconstruction_loss_tracker = keras.metrics.Mean(
			name="reconstruction_loss"
		)
		self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

	@property
	def metrics(self):
		return [
			self.total_loss_tracker,
			self.reconstruction_loss_tracker,
			self.kl_loss_tracker,
		]

	def train_step(self, data):
		with tf.GradientTape() as tape:

			mean, log_var, z = self.encoder(data)

			reconstruction = self.decoder(z)

			reconstruction_loss = tf.reduce_mean(
				tf.reduce_sum(
					keras.losses.binary_crossentropy(data, reconstruction),
					#axis=(1, 2),
				)
			)

			kl_loss = -0.5 * (1 + log_var - tf.square(mean) - tf.exp(log_var))
			kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
			total_loss = reconstruction_loss + kl_loss
		grads = tape.gradient(total_loss, self.trainable_weights)
		self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
		self.total_loss_tracker.update_state(total_loss)
		self.reconstruction_loss_tracker.update_state(reconstruction_loss)
		self.kl_loss_tracker.update_state(kl_loss)
		return {
			"loss": self.total_loss_tracker.result(),
			"reconstruction_loss": self.reconstruction_loss_tracker.result(),
			"kl_loss": self.kl_loss_tracker.result(),
		}

def vae_training(data, model_filepath, latent_dim, intermediate_dim1, intermediate_dim2, epochs, batch_size):

	x_train = np.array(data)
	input_dim = len(x_train[0])

	encoder_inputs = keras.Input(shape=(input_dim,))

	h1 = layers.Dense(intermediate_dim1, activation='relu')(encoder_inputs)
	h = layers.Dense(intermediate_dim2, activation='relu')(h1)
	mean = layers.Dense(latent_dim)(h)

	log_var = layers.Dense(latent_dim)(h)

	z = Sampling()([mean, log_var])
	encoder = keras.Model(encoder_inputs, [mean, log_var, z], name="encoder")
	encoder.summary()


	decoder_h1 = layers.Dense(intermediate_dim2, activation='relu')
	decoder_h = layers.Dense(intermediate_dim1, activation='relu')
	decoder_mean = layers.Dense(input_dim)
	decoder_intermediate = decoder_h1(z)
	h_decoded = decoder_h(decoder_intermediate)
	decoder_outputs = decoder_mean(h_decoded)

	decoder = keras.Model(z, decoder_outputs, name="decoder")
	decoder.summary()

	vae = VAE(encoder, decoder)
	vae.compile(optimizer=keras.optimizers.Adam())
	vae.fit(x_train, epochs=epochs, batch_size=batch_size)

	result = encoder.predict(x_train)

	encoder.save(model_filepath)

	return result, vae, encoder

def classify_feature(filepath, features):
	vae_autoencoder = keras.models.load_model(filepath)
	data_entry = np.vstack([features])
	predicted = vae_autoencoder.predict(data_entry)

	return predicted

def encoder_mapping(encoder, features):
	data_entry = np.vstack([features])
	predicted = encoder.predict(data_entry)
	return predicted

def arbitrator(att_encoder, benign_encoder, dbscan, features):
	att_predicted = encoder_mapping(att_encoder, features)
	benign_predicted = encoder_mapping(benign_encoder, features)
	att_out = dbscan_classifier(dbscan, att_predicted)
	benign_out = dbscan_classifier(dbscan, benign_predicted)
	combined_risk = float((benign_out - att_out + 1) / 2 )
	return combined_risk

