from torch import nn
from utils.torch import reparameterize


class VAE(nn.Module):
    def __init__(self, image_size=96):
        super(VAE, self).__init__()

        self.input_units = image_size * image_size
        self.output_units = image_size * image_size

        self.encoder = nn.Sequential(
            nn.Linear(self.input_units, 400),
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(400, 20)
        self.fc2 = nn.Linear(400, 20)

        self.decoder = nn.Sequential(
            nn.Linear(20, 400),
            nn.ReLU(),
            nn.Linear(400, self.output_units),
            nn.Sigmoid()
        )

    def encode(self, x):
        h1 = self.encoder(x)
        return self.fc1(h1), self.fc2(h1)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, self.input_units))
        z = reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
