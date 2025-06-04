# Hebbian Deep Encoding for Image Clustering

This project is exploring new Hebbian methods to train stackable image encoders. Afterward, the learned image embeddings are visualized using UMAP + K-Means Clustering.

Below is a sample clustering of learned image embeddings from the <a href="https://huggingface.co/datasets/zh-plus/tiny-imagenet" target="_blank">Tiny ImageNet</a> dataset.

<img src="/v2_imagenet_kmeans/tinyimagenet_hebbian_cluster_plot.png" width="100%" />

## Dataset Preparation

Images were extracted from either a sprite sheet or folder (e.g., Tiny ImageNet). For the Pokémon dataset, we used a 30x30 grid (898 total) from a transparent PNG.

```python
sprites = load_spritesheet("pokemon_all_transparent.png", sprite_size=(96, 96), tile_size=(96, 96))
```

For Tiny ImageNet:

```python
dataset = ImageFolder("../data/tiny_imagenet/train", transform=transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor()
]))
```

## Hebbian Encoder Architecture

Each Hebbian encoder layer applies:

* A \$3 \times 3\$ convolution with stride 2
* ReLU activation
* L2 normalization over spatial dimensions
* Lateral weights updated by Hebbian rule:

$$
\Delta W_{ij} = \eta \cdot \langle a_i \cdot a_j \rangle
$$

Where:

* \$\Delta W\_{ij}\$ is the change in weight from unit \$j\$ to \$i\$
* \$\eta\$ is the learning rate
* \$a\_i\$ and \$a\_j\$ are the activations of units \$i\$ and \$j\$ respectively
* \$\langle a\_i \cdot a\_j \rangle\$ denotes the batch-averaged outer product

This update promotes co-activation patterns. Inhibition is enforced by subtracting mean activation:

```python
hebbian = torch.einsum("bni,bnj->nij", act_flat, act_flat)  # outer product over batch
delta = 0.001 * hebbian.mean(dim=0)
self.lateral_weights.data += delta.clamp(-1.0, 1.0)
```

### Energy, Delta, Norm Logging

During training, the following values are logged per step:

* **Energy**: mean squared activation across all units, i.e., \$\mathbb{E}\[|a|^2]\$
* **Delta**: mean absolute change in lateral weights
* **Norm**: Frobenius norm of the lateral weight matrix, i.e., \$|W|\_F\$

Example:

```
[LOG] Step 122: energy=0.1466, delta=0.000166, norm=155.7744
```

This gives insight into encoder dynamics: stable energy and delta values indicate convergence, while growing norm may suggest over-association.

## Feature Extraction

Images are passed through a multi-layer encoder consisting of 4 HebbianEncoder layers. The final feature map is flattened to a 1D vector and stored.

```python
features = model(images)
features = F.normalize(features, dim=1).cpu().numpy()
```

## Embedding and Clustering Visualization

UMAP is used to project feature vectors to 2D.

```python
reducer = UMAP(n_components=2, random_state=42)
reduced = reducer.fit_transform(features)
```

Optionally, KMeans clustering is applied to the feature space:

```python
kmeans = KMeans(n_clusters=6, random_state=42).fit(features)
```

These steps are primarily for visualization and evaluation. They allow us to inspect whether the encoder has organized inputs meaningfully—not as a training objective.

## Layout and Plotting

Projected coordinates are normalized to fit inside a square canvas (e.g., 2500x2500 pixels). Margin padding ensures images are not clipped.

```python
reduced -= reduced.min(axis=0)
reduced /= (reduced.max(axis=0) + 1e-8)
reduced *= (canvas_size - 2 * margin)
reduced += margin
```

Sprites are drawn onto the canvas using their corresponding (x, y) UMAP coordinates.

```python
for (x, y), img in zip(reduced, sprites):
    pil = transforms.ToPILImage()(img).resize((sprite_size, sprite_size))
    canvas.paste(pil, (int(x), int(y)), mask=pil if has_alpha else None)
```

## Results

Tiny ImageNet (Hebbian):

<img src="/v2_imagenet_kmeans/tinyimagenet_hebbian_cluster_plot.png" width="100%" />

Pokémon Full RGBA Hebbian:

<img src="/v2_pokemon_kmeans/pokemon_hebbian_rgba_large_cluster_plot.png" width="100%" />

Pokémon Similarity:

<img src="/v2_pokemon/pokemon_hebbian_deep_encode_neighbors_grid.png" width="100%" />

