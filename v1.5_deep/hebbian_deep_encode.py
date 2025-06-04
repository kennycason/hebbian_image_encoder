import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# === Config ===
IMAGE_DIR = "../data/tiny_imagenet/train"
SPRITE_SIZE = (64, 64)
BATCH_SIZE = 8
NUM_IMAGES = 1000
ROWS = 10
GRID_K = 5
EPOCHS = 5

# === Hebbian Encoder Layer with Normalization and Inhibition ===
class HebbianEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, spatial):
        super().__init__()
        self.encode = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        C = out_channels
        N = spatial[0] * spatial[1]
        self.lateral_weights = torch.nn.Parameter(torch.zeros(C, N, N))

    def forward(self, x, step=None):
        act = F.relu(self.encode(x))
        act = act / (act.norm(dim=(2, 3), keepdim=True) + 1e-6)  # normalize activations
        B, C, H, W = act.shape
        act_flat = act.view(B, C, -1)

        with torch.no_grad():
            hebbian = torch.einsum("bni,bnj->nij", act_flat, act_flat)
            delta = 0.001 * hebbian.mean(dim=0)
            self.lateral_weights.data += delta
            self.lateral_weights.data.clamp_(-1.0, 1.0)

        lateral = torch.einsum("bci,cij->bcj", act_flat, self.lateral_weights)
        lateral = lateral.view(B, C, H, W)
        lateral = lateral - lateral.mean(dim=(2, 3), keepdim=True)  # inhibition
        act += lateral

        if step is not None:
            print(f"[LOG] Step {step}: energy={act.pow(2).mean():.4f}, delta={delta.abs().mean():.6f}, norm={self.lateral_weights.data.norm():.4f}")

        return act

# === Deep Hebbian Network ===
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
    return subset, DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False)

# === Cosine Similarity ===
def cosine_similarity_matrix(features):
    norm = features / features.norm(dim=1, keepdim=True)
    return torch.mm(norm, norm.T).cpu().numpy()

# === Grid Builder ===
def build_neighbor_grid(dataset, similarity, ref_indices, k=5):
    all_imgs = [img for img, _ in dataset]
    grid = []
    for idx in ref_indices:
        row = [all_imgs[idx]]
        topk = similarity[idx].argsort()[::-1][1:k + 1]
        row += [all_imgs[i] for i in topk if i < len(all_imgs)]
        grid.append(row)

    grid_w = SPRITE_SIZE[0] * (k + 1)
    grid_h = SPRITE_SIZE[1] * len(ref_indices)
    canvas = Image.new("RGB", (grid_w, grid_h))

    for y, row in enumerate(grid):
        for x, img in enumerate(row):
            pil = transforms.ToPILImage()(img)
            canvas.paste(pil, (x * SPRITE_SIZE[0], y * SPRITE_SIZE[1]))

    canvas.save("hebbian_deep_encode_neighbors_grid.png")
    print("[SAVED] hebbian_deep_encode_neighbors_grid.png")

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
                all_features.append(z.detach())

    features = torch.cat(all_features, dim=0)
    sim = cosine_similarity_matrix(features)
    np.save("hebbian_deep_encode_similarity.npy", sim)

    plt.figure(figsize=(8, 8))
    plt.imshow(sim, cmap="magma")
    plt.title("Hebbian Deep Cosine Similarity")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("hebbian_deep_encode_similarity.png")
    print("[SAVED] hebbian_deep_encode_similarity.png")

    step = len(dataset) // ROWS
    refs = [i * step for i in range(ROWS)]
    build_neighbor_grid(dataset, sim, refs, k=GRID_K)