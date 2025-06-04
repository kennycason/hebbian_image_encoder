# Deep Hebbian Image Encoder

This project explores biologically-inspired Hebbian methods to train stackable image encoders.
These models can organize visual information without traditional loss functions or backpropagation. 
Afterward, the learned image embeddings are visualized using UMAP + K-Means Clustering.

Below is a sample clustering of learned image embeddings from the <a href="https://huggingface.co/datasets/zh-plus/tiny-imagenet" target="_blank">Tiny ImageNet</a> dataset.

<img src="/v2_imagenet_kmeans/tinyimagenet_hebbian_cluster_plot.png" width="100%" />

<a href="https://github.com/kennycason/hebbian_image_encoder" target="_blank">Source on GitHub</a>

## Why Hebbian? (Biological Analogy)

Unlike traditional neural networks that rely on error signals and gradient descent, Hebbian models operate more like biological brains. There’s no supervision, no target output, no error propagation. Just forward input, and strengthening connections between neurons that activate together.

> “Neurons that fire together, wire together.”

This means the network doesn’t know what the correct answer is, it just recognizes and reinforces co-occurrence in the data. If two features appear frequently at the same time, their connection strengthens. Over time, this results in internal representations that reflect the structure of the data, without needing labels or supervision.

This approach mirrors parts of the brain’s learning strategy, where connections are locally updated based on experience, rather than global goals. It’s simpler, and more biologically plausible, and that simplicity might be useful for efficient or modular systems in the future.

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

## Energy, Delta, Norm Logging

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

## Hebbian Network Structure

The Hebbian encoder processes 96×96 RGBA Pokémon sprites using a stack of convolutional layers with stride 2. 
Each layer halves the spatial resolution while increasing the channel count. 
Each Hebbian layer also includes lateral recurrent weights trained with Hebbian updates to reinforce co-activation patterns.

```python
model = MultiLayerHebbian([
    (4, 16, (48, 48)),
    (16, 32, (24, 24)),
    (32, 64, (12, 12)),
    (64, 128, (6, 6))
])
```

Each tuple in the list specifies the parameters for a `HebbianEncoder` layer:

```python
(in_channels, out_channels, spatial_shape)
```

This configuration maps as follows:

| Layer | Input Channels | Output Channels | Input Spatial Size | Output Spatial Size |
| ----- | -------------- | --------------- | ------------------ | ------------------- |
| 1     | 4 (RGBA)       | 16              | 96×96              | 48×48               |
| 2     | 16             | 32              | 48×48              | 24×24               |
| 3     | 32             | 64              | 24×24              | 12×12               |
| 4     | 64             | 128             | 12×12              | 6×6                 |

This structure results in a final feature tensor of shape `(B, 128, 6, 6)` per image, which is flattened to `(B, 4608)` and used for clustering and visualization.

The `spatial` argument in each `HebbianEncoder` is required to initialize the lateral weight matrix:

```python
self.lateral_weights = torch.nn.Parameter(torch.zeros(C, N, N))
```

where `N = H × W` (the number of spatial positions per channel). These weights are updated using Hebbian learning:

```math
\Delta W_{ij} = \eta \cdot \langle a_i \cdot a_j \rangle
```

which in code becomes:

```python
hebbian = torch.einsum("bni,bnj->nij", act_flat, act_flat)
delta = 0.001 * hebbian.mean(dim=0)
self.lateral_weights.data += delta
```

This configuration balances spatial compression and representational capacity, while the Hebbian lateral updates encourage neurons to specialize by detecting and reinforcing co-activation patterns.


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

## Code + Results

### Tiny ImageNet (Hebbian)

<a href="/v2_imagenet_kmeans/imagenet_hebbian_deep_encode.py" target="_blank">Source</a>

<img src="/v2_imagenet_kmeans/tinyimagenet_hebbian_cluster_plot.png" width="100%" />

<img src="/v2_imagenet_kmeans/tinyimagenet_hebbian_cluster_plot_v2.png" width="100%" />

<img src="/v2_imagenet_kmeans/tinyimagenet_hebbian_cluster_plot_v1.png" width="100%" />


### Pokémon Full RGBA Hebbian

<a href="/v2_pokemon_kmeans/pokemon_hebbian_deep_encode_rgba_large.py" target="_blank">Source</a>

<img src="/v2_pokemon_kmeans/pokemon_hebbian_rgba_large_cluster_plot.png" width="100%" />

<img src="/v2_pokemon_kmeans/pokemon_hebbian_cluster_plot_898.png" width="100%" />

<img src="/v2_pokemon_kmeans/pokemon_hebbian_rgba_cluster_plot_v1.png" width="100%" />


### Pokémon Similarity

The first column contains randomly selected Pokémon, and then the most-similar 5 Pokémon are listed to the right.

<a href="/v2_pokemon/pokemon_hebbian_deep_encode.py" target="_blank">Source</a>

<img src="/v2_pokemon/pokemon_hebbian_deep_encode_neighbors_grid.png" width="100%" />


### Hebbian Image Encoder (Single-Layer)

This was the first prototype. The first column contains randomly selected image (from Tiny Imagenet), and then the most-similar 5 images are listed to the right.

<a href="/v1/hebbian_encode.py" target="_blank">Source</a>

<img src="/v1/hebbian_encode_neighbors_grid.png" width="66%" />
