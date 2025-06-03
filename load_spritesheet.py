from PIL import Image
import torch
import torchvision.transforms as transforms

def load_spritesheet(image_path, sprite_size=(32, 32), tile_size=(96, 96), max_sprites=None):
    """
    Loads and resizes Pokémon sprites from a sprite sheet.

    Args:
        image_path (str): path to the sprite sheet image
        sprite_size (tuple): final (width, height) for resized sprite
        tile_size (tuple): (width, height) of each tile in the sheet
        max_sprites (int): maximum number of sprites to load

    Returns:
        torch.Tensor: Tensor of shape [N, C, H, W]
    """
    image = Image.open(image_path).convert('RGBA')

    tile_width, tile_height = tile_size
    num_cols = image.width // tile_width
    num_rows = image.height // tile_height

    transform = transforms.Compose([
        transforms.Resize(sprite_size),
        transforms.ToTensor(),
    ])

    sprites = []
    for row in range(num_rows):
        for col in range(num_cols):
            if max_sprites and len(sprites) >= max_sprites:
                break
            left = col * tile_width
            upper = row * tile_height
            right = left + tile_width
            lower = upper + tile_height
            sprite = image.crop((left, upper, right, lower))
            sprites.append(transform(sprite))
        if max_sprites and len(sprites) >= max_sprites:
            break

    return torch.stack(sprites)