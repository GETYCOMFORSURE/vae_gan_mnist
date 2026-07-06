# VAE Project Plan

## Goal
Implement a Variational Autoencoder in PyTorch from scratch, then apply it to molecular data.

## Phase 1 — Understand the math (2-3h)
- What is an autoencoder (encode → compress → decode)
- What makes it *variational* (latent space is a distribution, not a point)
- The loss function: reconstruction loss + KL divergence
- Why this matters for molecule generation

## Phase 2 — Implement on simple data (3-4h)
- Dataset: MNIST digits (standard VAE benchmark)
- Build encoder: input → mean and variance of latent distribution
- Build decoder: sample from latent space → reconstruct input
- Train and visualize: can we generate new digits by sampling?

## Phase 3 — Apply to molecules (3-4h)
- Dataset: SMILES strings (molecular representations)
- Encode molecules into continuous latent space
- Sample new points → decode into new molecules
- Compare to Gómez-Bombarelli et al. 2018 (the seminal paper)

## Expected outcome
A working VAE that can generate novel molecular structures by sampling from a learned latent space.
