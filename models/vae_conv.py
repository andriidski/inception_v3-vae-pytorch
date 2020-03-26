from torch import nn
from utils.torch import flatten, un_flatten, reparameterize


class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()

        self.encode = nn.Sequential(
            # 3 x 96 x 96 -> 3 x 96 x 96
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            # 3 x 96 x 96 -> 32 x 32 x 32
            nn.Conv2d(3, 32, kernel_size=5, stride=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 32 x 32 x 32 -> 64 x 10 x 10
            nn.Conv2d(32, 64, kernel_size=5, stride=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64 x 10 x 10 -> 128 x 4 x 4
            nn.Conv2d(64, 128, kernel_size=5, stride=3, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #
            # 128 x 4 x 4 -> 256 x 2 x 2
            # equivalent to 1024 x 1 x 1 when flattened
            #
            nn.Conv2d(128, 256, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # fully-connected and latent space
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)

        self.decode = nn.Sequential(
            # 256 x 2 x 2 -> 128 x 4 x 4
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 128 x 4 x 4 -> 64 x 10 x 10
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=3, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64 x 10 x 10 -> 32 x 32 x 32
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 32 x 32 x 32 -> 3 x 96 x 96
            nn.ConvTranspose2d(32, 3, kernel_size=5, stride=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            # 3 x 96 x 96 -> 3 x 96 x 96
            nn.ConvTranspose2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )

    def bottleneck(self, h):
        mu, log_var = self.fc1(h), self.fc2(h)
        z = reparameterize(mu, log_var)
        return z, mu, log_var

    def encoder(self, x):
        h = flatten(self.encode(x))
        z, mu, log_var = self.bottleneck(h)
        return z, mu, log_var

    def decoder(self, z):
        z = self.fc3(z)
        z = self.decode(un_flatten(z, channels=256, h=2, w=2))
        return z

    def forward(self, x):
        z, mu, log_var = self.encoder(x)
        z = self.decoder(z)
        return z, mu, log_var
