This is the source code for the algorithms in the variational autoencoder (VAE) based DDoS attack defense

The VAE algorithm is in vae.py

The dbscan algorithm is in the dbscan.py

The actor-critic algorithm is in the actor.py

To train the VAE, load the data into python list data then call 

_, vae, encoder = vae_training(data, file_path, latent_dim, intermediate_dim1, intermediate_dim2, epochs, batch_size)


here file_path is the file path to save the trained model, latent_dim, intermediate_dim1, intermediate_dim2 are the 
number of latent dimension, intermediate dimension 1, intermediate dimension 2 resectively. epochs is the number of 
of epochs, batch_size is the training batch size. 

This training function wroks for both benign and attack autoencoders

To classify a certain feature (in python list) , call

encoder.predict(feature)

    
