import glob
import os
from typing import cast

from PIL import Image


def tile_images(input_folder, output_path, num_rows=2):
    """
    Tile images from a folder into a grid with a specified number of rows.

    Args:
        input_folder: Path to folder containing images
        output_path: Path to save the tiled image
        num_rows: Number of rows in the grid
    """
    # Get all empty_*.png and box_*.png images
    empty_images = sorted(glob.glob(os.path.join(input_folder, "empty_*.png")))
    box_images = sorted(glob.glob(os.path.join(input_folder, "box_*.png")))

    # Combine all images
    all_images = empty_images + box_images

    if not all_images:
        print(f"No images found in {input_folder}")
        return

    # Calculate grid dimensions
    num_images = len(all_images)
    num_cols = (num_images + num_rows - 1) // num_rows  # Ceiling division

    # Load the first image to get dimensions
    first_img = Image.open(all_images[0])
    img_width, img_height = first_img.size

    # Create a new image to hold the grid
    grid_width = num_cols * img_width
    grid_height = num_rows * img_height
    grid_img = Image.new("RGBA", (grid_width, grid_height), (255, 255, 255, 0))

    # Place each image in the grid
    for i, img_path in enumerate(all_images):
        img = Image.open(img_path)
        row = i // num_cols
        col = i % num_cols
        x = col * img_width
        y = row * img_height
        grid_img.paste(img, (x, y))
        print(f"Placed {os.path.basename(img_path)} at position ({row}, {col})")

    # Save the result
    # grid_img.show()
    grid_img.save(output_path)
    print(f"Created tiled image at {output_path}")


def overlay_images(image_paths, output_path):
    """
    Overlay multiple images with increasing opacity.
    The first image will be most transparent, the last will be most opaque.

    Args:
        image_paths: List of paths to images to overlay
        output_path: Path to save the overlaid image
    """
    if not image_paths:
        print("No images to overlay")
        return

    # Open the first image to get dimensions and use as base
    base_img = Image.open(image_paths[0]).convert("RGBA")

    if len(image_paths) == 1:
        base_img.save(output_path)
        print(f"Only one image found, saved to {output_path}")
        return

    # Calculate alpha step based on the number of images
    # Max alpha is 255, but we'll keep a bit of transparency even for the last image
    max_alpha = 255
    min_alpha = 50
    if len(image_paths) > 1:
        alpha_step = (max_alpha - min_alpha) / (len(image_paths) - 1)
    else:
        alpha_step = 0

    # Create a new composite image
    composite = Image.new("RGBA", base_img.size, (255, 255, 255, 255))
    # base_img.putalpha(200)
    composite = Image.alpha_composite(composite, base_img)

    # Overlay each image with appropriate alpha
    for i, img_path in enumerate(image_paths[1:]):
        # Calculate alpha for this layer (increasing with each image)
        alpha = int(min_alpha + i * alpha_step)

        # Open image and convert to RGBA
        img = Image.open(img_path).convert("RGBA")

        # Make a copy of the image with adjusted alpha
        img_with_alpha = Image.new("RGBA", img.size, (0, 0, 0, 0))

        # For each pixel, adjust alpha channel
        img_data = img.getdata()
        new_data = []

        for item in cast("list[tuple[int, int, int, int]]", img_data):
            # Keep original RGB, but scale the alpha
            if item[3] > 0:  # If the pixel is not completely transparent
                new_alpha = min(alpha, item[3])  # Don't increase alpha beyond original
                new_data.append((item[0], item[1], item[2], new_alpha))
            else:
                new_data.append(item)  # Keep completely transparent pixels as is

        img_with_alpha.putdata(new_data)

        # Composite this layer onto the result
        composite = Image.alpha_composite(composite, img_with_alpha)
        print(f"Added {os.path.basename(img_path)} with alpha {alpha}")

    # resize
    composite = composite.resize(
        (int(0.75 * composite.width), int(0.75 * composite.height))
    )

    # square crop
    width, height = composite.size
    crop_amount = int(width * 0.15)

    left = crop_amount
    top = 0
    right = width - crop_amount
    bottom = height

    composite = composite.crop((left, top, right, bottom))

    # Save the result
    composite.show()
    composite.save(output_path)
    print(f"Created overlay image at {output_path}")


if __name__ == "__main__":
    # Path to your images folder
    input_folder = "data/images/hardware_bg/"

    # Paths to save the output images
    output_tiled_path = "tiled_hardware_images.png"
    output_empty_overlay_path = "overlay_empty_images.png"
    output_box_overlay_path = "overlay_box_images.png"

    # Get all empty_*.png and box_*.png images
    empty_images = sorted(glob.glob(os.path.join(input_folder, "empty_*.png")))
    box_images = sorted(glob.glob(os.path.join(input_folder, "box_*.png")))

    # Create tiled image with 2 rows
    # tile_images(input_folder, output_tiled_path, num_rows=2)

    # Create overlay images
    # overlay_images(empty_images, output_empty_overlay_path)
    overlay_images(box_images, output_box_overlay_path)
