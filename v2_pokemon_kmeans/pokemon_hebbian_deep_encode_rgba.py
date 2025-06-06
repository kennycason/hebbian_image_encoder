import sys
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from umap import UMAP

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from load_spritesheet import load_spritesheet

# === Config ===
SPRITE_PATH = "../data/pokemon_all_transparent.png"
SPRITE_SIZE = (64, 64)
TILE_SIZE = (96, 96)
NUM_SPRITES = 30 * 30 - 2
NUM_IMAGES = 30 * 30 - 2
BATCH_SIZE = 8
EPOCHS = 50
CLUSTERS = 4
EMBED_SIZE = 64  # increased sprite size
CANVAS_SIZE = 1000

# === Hebbian Network ===
class HebbianEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, spatial):
        super().__init__()
        self.encode = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        C = out_channels
        N = spatial[0] * spatial[1]
        self.lateral_weights = torch.nn.Parameter(torch.zeros(C, N, N))

    def forward(self, x, step=None):
        act = F.relu(self.encode(x))
        act = act / (act.norm(dim=(2, 3), keepdim=True) + 1e-6)
        B, C, H, W = act.shape
        act_flat = act.view(B, C, -1)

        with torch.no_grad():
            hebbian = torch.einsum("bni,bnj->nij", act_flat, act_flat)
            delta = 0.001 * hebbian.mean(dim=0)
            self.lateral_weights.data += delta
            self.lateral_weights.data.clamp_(-1.0, 1.0)

        lateral = torch.einsum("bci,cij->bcj", act_flat, self.lateral_weights)
        lateral = lateral.view(B, C, H, W)
        lateral = lateral - lateral.mean(dim=(2, 3), keepdim=True)
        act += lateral

        if step is not None:
            print(f"[LOG] Step {step}: energy={act.pow(2).mean():.4f}, delta={delta.abs().mean():.6f}, norm={self.lateral_weights.data.norm():.4f}")

        return act

class MultiLayerHebbian(torch.nn.Module):
    def __init__(self, layer_shapes):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            HebbianEncoder(in_c, out_c, spatial) for (in_c, out_c, spatial) in layer_shapes
        ])

    def forward(self, x, step=None):
        for i, layer in enumerate(self.layers):
            x = layer(x, step=step if i == len(self.layers) - 1 else None)
        return x.view(x.size(0), -1).detach()

# === Load Sprites ===
def load_pokemon():
    sprites = load_spritesheet(SPRITE_PATH, sprite_size=SPRITE_SIZE, tile_size=TILE_SIZE, max_sprites=NUM_SPRITES)
    return sprites[:NUM_IMAGES]  # expect shape: (N, 4, H, W)

# === Plotting Utility ===
def plot_with_images(embeddings, images, size=EMBED_SIZE):
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='black', dpi=100)
    ax.set_facecolor('black')
    ax.set_xticks([])
    ax.set_yticks([])

    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    for (x, y), img_tensor in zip(embeddings, images):
        img = transforms.ToPILImage()(img_tensor).resize((size, size), resample=Image.BILINEAR)
        imbox = OffsetImage(img, zoom=1)
        ab = AnnotationBbox(imbox, (x, y), frameon=False)
        ax.add_artist(ab)

    ax.set_xlim(0, CANVAS_SIZE)
    ax.set_ylim(0, CANVAS_SIZE)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("pokemon_hebbian_rgba_cluster_plot.png", facecolor='black')
    print("[SAVED] pokemon_hebbian_rgba_cluster_plot.png")

# === Main ===
if __name__ == "__main__":
    sprites = load_pokemon()  # (N, 4, H, W)
    dataset = TensorDataset(sprites)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = MultiLayerHebbian([
        (4, 16, (32, 32)),
        (16, 32, (16, 16)),
        (32, 64, (8, 8)),
        (64, 128, (4, 4))
    ])

    all_features = []
    for epoch in range(EPOCHS):
        for step, (batch,) in enumerate(dataloader):
            z = model(batch, step=step) if epoch == EPOCHS - 1 else model(batch)
            if epoch == EPOCHS - 1:
                all_features.append(z)

    features = torch.cat(all_features, dim=0).cpu().numpy()
    features = np.nan_to_num(features)
    features /= (np.linalg.norm(features, axis=1, keepdims=True) + 1e-6)

    reducer = UMAP(n_components=2, random_state=42)
    reduced = reducer.fit_transform(features)

    margin = EMBED_SIZE // 2
    reduced -= reduced.min(axis=0)
    reduced /= (reduced.max(axis=0) + 1e-8)
    reduced *= (CANVAS_SIZE - 2 * margin)
    reduced += margin

    plot_with_images(reduced, sprites, size=EMBED_SIZE)