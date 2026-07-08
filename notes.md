# Notes 
## Autoencoder → VAE
### [variational autoencoders by arxiv insights](https://www.youtube.com/watch?v=9zKuYvjFFS8)
get intuition for autoencoders (not variational here)

<img width="893" height="768" alt="Screenshot 2026-07-08 at 10 47 42" src="https://github.com/user-attachments/assets/36feacfa-a414-4ca4-84e1-4d102e21eddd" />

### [Intuitively Understanding Variational Autoencoders](https://medium.com/data-science/intuitively-understanding-variational-autoencoders-1bfe67eb5daf)

variational autoencoders

<img width="698" height="674" alt="Screenshot 2026-07-08 at 12 47 25" src="https://github.com/user-attachments/assets/06b75205-9373-42a0-ae82-28a768d63942" />

variational sampler code

```python
# build your encoder upto here. It can simply be a series of dense layers, a convolutional network
# or even an LSTM decoder. Once made, flatten out the final layer of the encoder, call it hidden.
# we use Keras to build the graph
latent_size = 5
mean = Dense(latent_size)(hidden)
# we usually don't directly compute the stddev σ 
# but the log of the stddev instead, which is log(σ)
# the reasoning is similar to why we use softmax, instead of directly outputting
# numbers in fixed range [0, 1], the network can output a wider range of numbers which we can later compress down
log_stddev = Dense(latent_size)(hidden)
def sampler(mean, log_stddev):
    # we sample from the standard normal a matrix of batch_size * latent_size (taking into account minibatches)
    std_norm = K.random_normal(shape=(K.shape(mean)[0], latent_size), mean=0, stddev=1)
    # sampling from Z~N(μ, σ^2) is the same as sampling from μ + σX, X~N(0,1)
    return mean + K.exp(log_stddev) * std_norm
```
