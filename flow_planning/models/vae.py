import torch
import torch.nn as nn


class GECO:
    def __init__(
        self,
        goal=0.5,
        step_size=1e-4,
        alpha=0.99,
        beta_init=1.0,
        beta_min=1e-5,
        beta_max=1e5,
    ):
        self.goal = goal  # Target reconstruction error
        self.step_size = step_size  # Step size for beta adjustment
        self.alpha = alpha  # EMA coefficient
        self.beta = beta_init
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.ema_error = None

    def update(self, recon_error):
        # Initialize EMA if first update
        if self.ema_error is None:
            self.ema_error = recon_error

        # Update EMA of reconstruction error
        self.ema_error = self.alpha * self.ema_error + (1 - self.alpha) * recon_error

        # Compute constraint value: C(x) - goal
        constraint = self.goal - self.ema_error

        # Update beta using gradient ascent on the Lagrangian
        self.beta = self.beta * torch.exp(self.step_size * constraint)

        # Clamp beta to prevent extreme values
        self.beta = torch.clamp(self.beta, self.beta_min, self.beta_max)

        return self.beta, self.ema_error


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=32, hidden_dim=256, device="cuda"):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, return_latent=False):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)

        if return_latent:
            return x_recon, mu, logvar, z
        return x_recon, mu, logvar
