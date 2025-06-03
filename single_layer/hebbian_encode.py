# hebbian_encoder_test.py

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image

# === Config ===
IMAGE_DIR = "tiny_imagenet/train"
SPRITE_SIZE = (64, 64)
BATCH_SIZE = 8
NUM_IMAGES = 1000
ROWS = 10      # number of reference images (rows in final grid)
GRID_K = 5     # number of neighbors per reference
EPOCHS = 5

# === Hebbian Encoder Layer ===
class HebbianEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, spatial):
        super().__init__()
        self.encode = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        C = out_channels
        N = spatial[0] * spatial[1]
        self.lateral_weights = torch.nn.Parameter(torch.zeros(C, N, N))

    def forward(self, x, step=None):
        act = F.relu(self.encode(x))
        B, C, H, W = act.shape
        act_flat = act.view(B, C, -1)

        with torch.no_grad():
            hebbian = torch.einsum('bni,bnj->nij', act_flat, act_flat)
            delta = 0.001 * hebbian.mean(dim=0)
            self.lateral_weights.data += delta
            self.lateral_weights.data.clamp_(-1.0, 1.0)

        lateral = torch.einsum('bci,cij->bcj', act_flat, self.lateral_weights)
        act += lateral.view(B, C, H, W)

        # Logging
        if step is not None:
            act_energy = act.pow(2).mean().item()
            delta_magnitude = delta.abs().mean().item()
            weight_norm = self.lateral_weights.data.norm().item()
            print(f"[LOG] Step {step}: energy={act_energy:.4f}, delta={delta_magnitude:.6f}, norm={weight_norm:.4f}")

        return act.view(B, -1).detach()

# === Load ImageNet Images ===
def load_dataset():
    transform = transforms.Compose([
        transforms.Resize(SPRITE_SIZE),
        transforms.ToTensor()
    ])
    dataset = ImageFolder(IMAGE_DIR, transform=transform)
    subset = torch.utils.data.Subset(dataset, list(range(min(NUM_IMAGES, len(dataset)))))
    return subset, DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False)

# === Similarity Heatmap ===
def cosine_similarity_matrix(features):
    norm = features / features.norm(dim=1, keepdim=True)
    return torch.mm(norm, norm.T).cpu().numpy()

# === Build similarity image grid ===
def build_neighbor_grid(dataset, similarity, ref_indices, k=5):
    all_imgs = [img for img, _ in dataset]
    grid = []
    for idx in ref_indices:
        row = [all_imgs[idx]]
        topk = similarity[idx].argsort()[::-1][1:k + 1]
        topk = [i for i in topk if i < len(all_imgs)]
        row += [all_imgs[i] for i in topk]
        grid.append(row)

    grid_w = SPRITE_SIZE[0] * (k + 1)
    grid_h = SPRITE_SIZE[1] * len(ref_indices)
    canvas = Image.new("RGB", (grid_w, grid_h))

    for y, row in enumerate(grid):
        for x, img in enumerate(row):
            pil = transforms.ToPILImage()(img)
            canvas.paste(pil, (x * SPRITE_SIZE[0], y * SPRITE_SIZE[1]))

    canvas.save("hebbian_encode_neighbors_grid.png")
    print("[SAVED] hebbian_encode_neighbors_grid.png")

if __name__ == "__main__":
    dataset, dataloader = load_dataset()
    model = HebbianEncoder(in_channels=3, out_channels=16, spatial=(32, 32))

    all_features = []
    for epoch in range(EPOCHS):
        if epoch == EPOCHS - 1:
            for step, (batch, _) in enumerate(dataloader):
                z = model(batch, step=step)
                all_features.append(z)
        else:
            for step, (batch, _) in enumerate(dataloader):
                model(batch)

    features = torch.cat(all_features, dim=0)

    sim = cosine_similarity_matrix(features)
    np.save("hebbian_encode_similarity.npy", sim)
    plt.figure(figsize=(8, 8))
    plt.imshow(sim, cmap="magma")
    plt.title("Hebbian Encoder Cosine Similarity")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("hebbian_encode_similarity.png")
    print("[SAVED] hebbian_encode_similarity.png")

    step = len(dataset) // ROWS
    refs = [i * step for i in range(ROWS)]
    build_neighbor_grid(dataset, sim, refs, k=GRID_K)
