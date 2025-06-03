import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from load_spritesheet import load_spritesheet
from torchvision import transforms

# === Config ===
SPRITE_PATH = "data/pokemon_all.png"
SPRITE_SIZE = (64, 64)
TILE_SIZE = (96, 96)
NUM_SPRITES = 32
BATCH_SIZE = 8
NUM_IMAGES = 32
ROWS = 10
GRID_K = 5
EPOCHS = 30

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

        if step is not None:
            act_energy = act.pow(2).mean().item()
            delta_magnitude = delta.abs().mean().item()
            weight_norm = self.lateral_weights.data.norm().item()
            print(f"[LOG] Step {step}: energy={act_energy:.4f}, delta={delta_magnitude:.6f}, norm={weight_norm:.4f}")

        return act  # ← Keep full shape!

class MultiLayerHebbian(torch.nn.Module):
    def __init__(self, layer_shapes):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            HebbianEncoder(in_c, out_c, spatial)
            for (in_c, out_c, spatial) in layer_shapes
        ])

    def forward(self, x, step=None):
        for i, layer in enumerate(self.layers):
            x = layer(x, step=step if i == len(self.layers) - 1 else None)
        B = x.shape[0]
        return x.view(B, -1).detach()  # Flatten only once

def cosine_similarity_matrix(features):
    norm = features / features.norm(dim=1, keepdim=True)
    return torch.mm(norm, norm.T).cpu().numpy()

def build_neighbor_grid(images, similarity, ref_indices, k=5):
    grid = []
    for idx in ref_indices:
        row = [images[idx]]
        topk = similarity[idx].argsort()[::-1][1:k + 1]
        topk = [i for i in topk if i < len(images)]
        row += [images[i] for i in topk]
        grid.append(row)

    grid_w = SPRITE_SIZE[0] * (k + 1)
    grid_h = SPRITE_SIZE[1] * len(ref_indices)
    canvas = Image.new("RGB", (grid_w, grid_h))

    for y, row in enumerate(grid):
        for x, img in enumerate(row):
            pil = transforms.ToPILImage()(img)
            canvas.paste(pil, (x * SPRITE_SIZE[0], y * SPRITE_SIZE[1]))

    canvas.save("pokemon_hebbian_deep_encode_neighbors_grid.png")
    print("[SAVED] pokemon_hebbian_deep_encode_neighbors_grid.png")

if __name__ == "__main__":
    sprites = load_spritesheet(SPRITE_PATH, sprite_size=SPRITE_SIZE, tile_size=TILE_SIZE, max_sprites=NUM_SPRITES)[:, :3, :, :] # RGBA -> RGB
    sprites = sprites[:NUM_IMAGES]
    dataset = TensorDataset(sprites)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = MultiLayerHebbian([
        (3, 8, (32, 32)),
        (8, 16, (16, 16)),
        (16, 32, (8, 8)),
        (32, 64, (4, 4))
    ])

    all_features = []
    for epoch in range(EPOCHS):
        for step, (batch,) in enumerate(dataloader):
            z = model(batch, step=step)
            if epoch == EPOCHS - 1:
                all_features.append(z)

    features = torch.cat(all_features, dim=0)

    sim = cosine_similarity_matrix(features)
    np.save("pokemon_hebbian_deep_encode_similarity.npy", sim)
    plt.figure(figsize=(8, 8))
    plt.imshow(sim, cmap="magma")
    plt.title("Hebbian Deep Cosine Similarity")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("pokemon_hebbian_deep_encode_similarity.png")
    print("[SAVED] pokemon_hebbian_deep_encode_similarity.png")

    step = len(sprites) // ROWS
    refs = [i * step for i in range(ROWS)]
    build_neighbor_grid(sprites, sim, refs, k=GRID_K)