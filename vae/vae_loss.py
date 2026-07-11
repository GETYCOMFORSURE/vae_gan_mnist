# reconstruction loss + KL loss

# loss
images = images.view(images.size(0), -1)  # flatten [batch, 1, 28, 28] → [batch, 784]

mu, logvar = encoder(images)
z = sampler(mu, logvar)
reconstructed = decoder(z)

criterion = nn.BCELoss()
rec_loss = criterion(reconstructed, images)
kl_loss = -0.5 * torch.sum(1 + logvar - torch.square(mu) - torch.square(torch.exp(logvar)))
loss = rec_loss + kl_loss

# optimizer (corresponds to update parameters in numpy version)
encoder = Encoder()
decoder = Decoder()

optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.01)
