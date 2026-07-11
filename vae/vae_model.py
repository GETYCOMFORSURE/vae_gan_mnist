# VAE: encoder → μ, σ → sample once (z = μ + σ * ε) → decoder

# encoder outputs μ, σ
class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28*28, 200)     # shared layer: input → hidden
        self.relu = nn.ReLU()
        self.fc_mu = nn.Linear(200, 10)      # branch 1: hidden → mean
        self.fc_logvar = nn.Linear(200, 10)  # branch 2: hidden → log-variance

    def forward(self, x):
        h = self.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

# sampled stochastically
def sampler(mu, logvar):
    eps = torch.randn_like(mu)     # ε ~ N(0,1), same shape as mu
    std = torch.exp(0.5 * logvar)  # convert logvar → std
    z = mu + std * eps
    return z

# decoder
# two layers: ReLU in between, sigmoid at the end. 
class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(10, 200)   
        self.fc2 = nn.Linear(200, 28*28)   
        self.sig = nn.Sigmoid()
    
    def forward(self, y):
        h = self.relu(self.fc1(y))
        a = self.fc2(h)
        b = self.sig(a)
        return b
