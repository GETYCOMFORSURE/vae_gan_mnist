# VAE vs GAN Project Plan

## Goal
Implement both a VAE and a GAN in PyTorch on MNIST, then compare the two generative approaches.

## Data
MNIST (handwritten digits, 28×28 grayscale) via `torchvision.datasets.MNIST`.

## Phase 1 — Understand the math
- Autoencoder → VAE: latent space as a distribution, reconstruction loss + KL divergence
- GAN: Generator vs Discriminator, the minimax game, adversarial loss
- Key difference: VAE has an explicit likelihood objective, GAN doesn't
- Why GANs are harder to train (mode collapse, instability) vs why VAEs tend to produce blurrier samples

## Phase 2 — Implement VAE
- Encoder → mean/variance of latent distribution
- Decoder → sample from latent space → reconstruct
- Train and visualize: generate new digits by sampling

## Phase 3 — Implement GAN
- Generator: latent noise → fake digit
- Discriminator: image → real/fake
- Train both alternately, track loss curves

## Phase 4 — Direct comparison
- Sample quality: VAE (smooth, blurrier) vs GAN (sharper, riskier)
- Training stability: VAE loss curve vs GAN loss curve — does GAN show collapse/oscillation?
- Latent space smoothness: VAE's is built-in; does GAN's interpolate as cleanly?
- Write-up: GAN's advantages, and where its problems actually show up — answering Cheng's question directly
