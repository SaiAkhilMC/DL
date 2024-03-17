import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

torch.set_printoptions(precision=2, sci_mode=None)
np.set_printoptions(precision=2, suppress=True)

torch.manual_seed(42)

transform = transforms.Compose([transforms.ToTensor()])

batch_size = 64
train_loader = DataLoader(datasets.CIFAR10('.', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i in range(data.size()[0]):
    ax = axes[i // 4, i % 4]
    image = data[i].permute(1, 2, 0).numpy()
    ax.imshow(image)
    ax.axis('off')

plt.show()

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(128*4*4, latent_dim)
        self.fc_logvar = nn.Linear(128*4*4, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128*4*4),
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld

latent_dim = 32
epochs = 2
learning_rate = 0.001

vae = VAE(latent_dim)
print("Variational Autoencoder architecture: \n", vae, "\n")

optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

from torchsummary import summary
print("Model Summary: \n\n", summary(vae, (3, 32, 32)), "\n")

vae.train()

import time

start_time = time.time()

for epoch in range(epochs):
    total_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        recon_data, mu, logvar = vae(data)
        loss = vae_loss(recon_data, data, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader.dataset):.4f}')

end_time = time.time()
total_training_time = end_time - start_time
print("Total Training time: \t", (total_training_time), " minutes. \n")

from google.colab import drive
drive.mount('/content/drive')

location = "/content/drive/My Drive"
torch.save(vae.state_dict(), location+"vae_cifar10.pth")

vae.eval()

with torch.no_grad():
    sample = torch.randn(16, latent_dim)
    generated_samples = vae.decode(sample)
    generated_samples = generated_samples.permute(0, 2, 3, 1).numpy()

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(16):
        ax = axes[i // 4, i % 4]
        ax.imshow(generated_samples[i])
        ax.axis('off')

    plt.show()
