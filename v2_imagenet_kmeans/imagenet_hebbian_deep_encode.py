import sys
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from umap import UMAP

# === Config ===
IMAGE_DIR = "../data/tiny_imagenet/train"
SPRITE_SIZE = (64, 64)
BATCH_SIZE = 8
NUM_IMAGES = 1000
EPOCHS = 50
CLUSTERS = 4
EMBED_SIZE = 32  # thumbnail size
CANVAS_SIZE = 2500

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

# === Load Dataset ===
def load_dataset():
    transform = transforms.Compose([
        transforms.Resize(SPRITE_SIZE),
        transforms.ToTensor()
    ])
    dataset = ImageFolder(IMAGE_DIR, transform=transform)
    subset = torch.utils.data.Subset(dataset, list(range(min(NUM_IMAGES, len(dataset)))))
    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False)
    return subset, loader

# === Plot Utility ===
def plot_with_images(embeddings, images, title="Hebbian Clusters", size=32, canvas_size=2500):
    fig, ax = plt.subplots(figsize=(canvas_size / 100, canvas_size / 100), facecolor='black', dpi=100)
    ax.set_facecolor('black')
    ax.set_title(title, color='white')
    ax.set_xticks([])
    ax.set_yticks([])

    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    # Normalize coordinates to canvas
    margin = size * 2
    embeddings -= embeddings.min(axis=0)
    embeddings /= (embeddings.max(axis=0) + 1e-8)
    embeddings *= (canvas_size - 2 * margin)
    embeddings += margin

    for (x, y), img_tensor in zip(embeddings, images):
        img = transforms.ToPILImage()(img_tensor).resize((size, size), resample=Image.BILINEAR).convert("RGB")
        imbox = OffsetImage(img, zoom=1.5)  # zoom factor for visibility
        ab = AnnotationBbox(imbox, (x, y), frameon=False)
        ax.add_artist(ab)

    ax.set_xlim(0, canvas_size)
    ax.set_ylim(0, canvas_size)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("tinyimagenet_hebbian_cluster_plot.png", facecolor='black')
    print("[SAVED] tinyimagenet_hebbian_cluster_plot.png")

# === Main ===
if __name__ == "__main__":
    dataset, dataloader = load_dataset()
    model = MultiLayerHebbian([
        (3, 16, (32, 32)),
        (16, 32, (16, 16)),
        (32, 64, (8, 8)),
        (64, 128, (4, 4))
    ])

    all_features = []
    for epoch in range(EPOCHS):
        for step, (batch, _) in enumerate(dataloader):
            z = model(batch, step=step) if epoch == EPOCHS - 1 else model(batch)
            if epoch == EPOCHS - 1:
                all_features.append(z)

    features = torch.cat(all_features, dim=0).cpu().numpy()
    features = np.nan_to_num(features)
    features /= (np.linalg.norm(features, axis=1, keepdims=True) + 1e-6)

    reducer = UMAP(n_components=2, random_state=42) #, min_dist=0.2)
    reduced = reducer.fit_transform(features)

    margin = EMBED_SIZE // 2
    reduced -= reduced.min(axis=0)
    reduced /= (reduced.max(axis=0) + 1e-8)
    reduced *= (CANVAS_SIZE - 2 * margin)
    reduced += margin

    all_images = [img for img, _ in dataset]
    plot_with_images(reduced, all_images)