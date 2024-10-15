# %%
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.ndimage import affine_transform

# Add parent directory to sys.path
path_current = os.path.dirname(__file__)
path_parent = os.path.abspath(os.path.join(path_current, ".."))
sys.path.append(path_parent)

# Now import from utilities
from utilities import *


def normalize_image(image):
    image = 255 - image
    image = ((image / image.max()) * 255).round().astype(int)
    return image


class CellGenerator:

    def __init__(
        self,
        image_size=100,
        center_inner=0.5,
        center_outer=0.5,
        radius_inner=0.2,
        radius_outer=0.3,
        skew=0,
    ):
        """Note that center is given in normalized coordinates (between 0 and
        1), so that (0, 1) indicates the upper left corner."""
        self.image_size = image_size

        # Scale to the size of the image
        self.center_inner = np.array(center_inner) * image_size
        self.center_outer = np.array(center_outer) * image_size
        self.radius_inner = radius_inner * image_size
        self.radius_outer = radius_outer * image_size
        self.skew = skew

    def normalize(self, image):
        image = image / image.max()
        return image

    def skew_cell(self, image, skew):
        skew_matrix = np.array(
            [
                [1, skew, 0],  # Horizontal skew (second element of the first row)
                [0, 1, 0],  # Keep the y-axis scaling (identity)
                [0, 0, 1],
            ]
        )
        return affine_transform(image, skew_matrix[:2, :2])

    def generate_cell(self):
        inner_disc = self.generate_disc(self.radius_inner, self.center_inner)
        outer_disc = self.generate_disc(self.radius_outer, self.center_outer)
        cell_img = self.normalize(inner_disc + outer_disc)
        cell_img = self.skew_cell(cell_img, self.skew)
        return cell_img

    def generate_disc(self, radius, center):
        """
        Generate an image of a circle where pixels inside the circle are black (0)
        and pixels outside the circle are white (255).
        """
        # Create an empty white image (255 for white)
        image = np.zeros((self.image_size, self.image_size))

        # Loop over every pixel in the image
        for i in range(self.image_size):
            for j in range(self.image_size):
                # Compute the distance from the center
                dist_from_center = ((i - center[0]) ** 2 + (j - center[1]) ** 2) ** 0.5

                # If the pixel is inside the circle, set it to black (0)
                if dist_from_center <= radius:
                    image[i, j] = 1.0

        return image


image_size = 100

skew_values = np.random.randn(3) * 0.2
centers_inner = np.random.uniform(size=(3, 2))
centers_outer = np.random.normal(centers_inner, scale=0.02)
radiuses_inner = np.random.uniform(0.1, 0.15, size=(3, 1))
radiuses_outer = np.random.uniform(radiuses_inner, radiuses_inner + 0.05)

cell1 = CellGenerator(
    image_size=image_size,
    center_inner=centers_inner[0],
    center_outer=centers_outer[0],
    radius_inner=radiuses_inner[0],
    radius_outer=radiuses_outer[0],
    skew=skew_values[0],
).generate_cell()
plt.imshow(cell1)

cell2 = CellGenerator(
    image_size=image_size,
    center_inner=(0.7, 0.9),
    center_outer=(0.69, 0.91),
    radius_inner=0.08,
    radius_outer=0.15,
    skew=skew_values[1],
).generate_cell()

cell3 = CellGenerator(
    image_size=image_size,
    center_inner=(0.4, 0.52),
    center_outer=(0.4, 0.53),
    radius_inner=0.14,
    radius_outer=0.17,
    skew=skew_values[2],
).generate_cell()

cell_tissue = cell1 + cell2 + cell3

# Display the generated circle image
fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(cell1, cmap="gray")
ax[0, 0].set_title(f"Cell1")

ax[0, 1].imshow(cell2, cmap="gray")
ax[0, 1].set_title(f"Cell2")

ax[1, 0].imshow(cell3, cmap="gray")
ax[1, 0].set_title(f"Cell3")

ax[1, 1].imshow(cell_tissue, cmap="gray")
ax[1, 1].set_title(f"Tissue")

# %% GENERATE RANDOM CELLS
image_size = 200
n_cells = 5

skew_values = np.random.randn(n_cells) * 0.2
centers_inner = np.random.uniform(size=(n_cells, 2))
centers_outer = np.random.normal(centers_inner, scale=0.001)
radiuses_inner = np.random.uniform(0.03, 0.05, size=(n_cells, 1))
radiuses_outer = np.random.uniform(radiuses_inner, radiuses_inner + 0.025)

cells = []

for i in range(n_cells):
    cell = CellGenerator(
        image_size=image_size,
        center_inner=centers_inner[i],
        center_outer=centers_outer[i],
        radius_inner=radiuses_inner[i],
        radius_outer=radiuses_outer[i],
        skew=skew_values[i],
    ).generate_cell()
    cells.append(cell)

tissue_rand_gen = np.array(cells).sum(axis=0)
plt.imshow(tissue_rand_gen)
