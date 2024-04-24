# Variational Autoencoder for HalfCheetah-v4 Environment

## Overview
This project implements a Variational Autoencoder (VAE) to model and predict the dynamics of the HalfCheetah-v4 environment from OpenAI's Gym library. It uses JAX for automatic differentiation and efficient computation.

## Model Description
The VAE comprises an encoder, transition model, and decoder:
- **Encoder**: Converts observations to a latent space.
- **Transition**: Predicts the next latent state based on the current one.
- **Decoder**: Maps latent states back to observations.
The model incorporates reparameterization for randomness and uses Variational Free Energy as the loss function, composed of KL divergence and Cross-Entropy (simplified to MSE under Gaussian assumptions).

## Dependencies
- numpy
- jax
- jaxlib
- matplotlib
- gymnasium

## Usage
To run the training:
```bash
python sample_main_loop_with_matrixes_as_weight.py
```
