import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from umap import UMAP
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from load_spritesheet import load_spritesheet

# === Config ===
SPRITE_PATH = "../data/pokemon_all_transparent.png"
SPRITE_SIZE = (96, 96)
TILE_SIZE = (96, 96)
NUM_SPRITES = 898
BATCH_SIZE = 16
NUM_IMAGES = 898
EPOCHS = 100
CLUSTERS = 10
EMBED_SIZE = 96
CANVAS_SIZE = 1000
LATENT_DIM = 128

# === Load Sprites ===
def load_pokemon():
    sprites = load_spritesheet(SPRITE_PATH, sprite_size=SPRITE_SIZE, tile_size=TILE_SIZE, max_sprites=NUM_SPRITES)
    return sprites[:NUM_IMAGES]

# === Autoencoder ===
class ConvEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 4, 2, 1),  # 96 -> 48
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), # 48 -> 24
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # 24 -> 12
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class ConvDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 12 * 12)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 12 -> 24
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 24 -> 48
            nn.ReLU(),
            nn.ConvTranspose2d(32, 4, 4, 2, 1),    # 48 -> 96
            nn.Sigmoid()  # assuming inputs in [0, 1]
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 128, 12, 12)
        return self.decoder(x)

class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z

# === Image Plotting Utility ===
def plot_with_images(embeddings, images, title="Autoencoder Clusters", size=EMBED_SIZE):
    fig, ax = plt.subplots(figsize=(25, 25), facecolor='black', dpi=100)
    ax.set_facecolor('black')
    ax.set_title(title, color='white')
    ax.set_xticks([])
    ax.set_yticks([])

    label_colors = plt.cm.tab10(np.linspace(0, 1, CLUSTERS))

    for i, ((x, y), img_tensor) in enumerate(zip(embeddings, images)):
        img = transforms.ToPILImage()(img_tensor).resize((size, size)).convert("RGBA")
        color = tuple((label_colors[kmeans.labels_[i]][:3] * 255).astype(np.uint8)) + (80,)  # semi-transparent
        overlay = Image.new("RGBA", img.size, color)
        img = Image.alpha_composite(img, overlay)
        imbox = OffsetImage(img, zoom=1)
        ab = AnnotationBbox(imbox, (x, y), frameon=False)
        ax.add_artist(ab)

    ax.set_xlim(0, CANVAS_SIZE)
    ax.set_ylim(0, CANVAS_SIZE)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("pokemon_autoencoder_cluster_plot.png", facecolor='black')
    print("[SAVED] pokemon_autoencoder_cluster_plot.png")

# === Main ===
if __name__ == "__main__":
    sprites = load_pokemon()
    dataset = TensorDataset(sprites)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Autoencoder(LATENT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training autoencoder...")
    for epoch in range(EPOCHS):
        for batch, in dataloader:
            out, _ = model(batch)
            loss = F.mse_loss(out, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"[Epoch {epoch:03d}] Loss = {loss.item():.6f}")

    # === Get Features ===
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_features = []
    with torch.no_grad():
        for batch, in dataloader:
            z = model.encoder(batch)
            all_features.append(z)

    features = torch.cat(all_features, dim=0).cpu().numpy()
    features = np.nan_to_num(features)
    features /= (np.linalg.norm(features, axis=1, keepdims=True) + 1e-6)

    # === Clustering + UMAP ===
    print("Clustering and projecting...")
    kmeans = KMeans(n_clusters=CLUSTERS, random_state=42).fit(features)
    reducer = UMAP(
        n_components=2,
        min_dist=0.01,
        spread=1.0,
        n_neighbors=20,
        random_state=42
    )
    reduced = reducer.fit_transform(features)

    margin = EMBED_SIZE // 2
    reduced -= reduced.min(axis=0)
    reduced /= (reduced.max(axis=0) + 1e-8)
    reduced *= (CANVAS_SIZE - 2 * margin)
    reduced += margin

    plot_with_images(reduced, sprites)