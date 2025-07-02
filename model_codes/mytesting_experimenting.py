import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def generate_irregular_3d_shape(depth, size=32, irregularity=0.3, growth_chance=0.6):
    """
    Generate a 3D matrix with a random irregular object.

    Args:
        depth (int): Number of slices along the z-axis.
        size (int): Dimensions of each 2D slice (size x size).
        irregularity (float): Controls the randomness in shape generation (0-1).
        growth_chance (float): Probability of growth in neighboring slices (0-1).

    Returns:
        numpy.ndarray: A 3D matrix with the generated shape.
    """
    # Initialize 3D space
    space = np.zeros((depth, size, size), dtype=np.float32)

    # Start with a random seed in the central slice
    center_slice = depth // 2
    seed_radius = size // 6
    y, x = np.ogrid[:size, :size]
    seed_center = (size // 2, size // 2)
    seed_mask = (x - seed_center[0]) ** 2 + (y - seed_center[1]) ** 2 <= seed_radius ** 2
    space[center_slice][seed_mask] = 1

    # Grow the object across slices
    for z in range(center_slice - 1, -1, -1):  # Grow upward
        previous_slice = space[z + 1]
        growth_mask = (
            (previous_slice > 0) & (np.random.rand(size, size) < growth_chance)
        )
        space[z] = np.clip(previous_slice + growth_mask, 0, 1)

    for z in range(center_slice + 1, depth):  # Grow downward
        previous_slice = space[z - 1]
        growth_mask = (
            (previous_slice > 0) & (np.random.rand(size, size) < growth_chance)
        )
        space[z] = np.clip(previous_slice + growth_mask, 0, 1)

    # Add irregularity by random erosion/dilation
    for z in range(depth):
        noise = np.random.rand(size, size)
        space[z] = np.where(noise < irregularity, 0, space[z])

    # Smooth the object to create more natural transitions
    space = gaussian_filter(space, sigma=1)

    # Normalize the matrix
    space /= np.max(space)

    return space

# Visualize the generated 3D shape
def visualize_3d_slices(matrix, title="Irregular 3D Object"):
    """
    Visualizes slices of a 3D matrix.
    """
    depth = matrix.shape[0]
    fig, axes = plt.subplots(1, depth, figsize=(depth * 2, 2))
    for i in range(depth):
        ax = axes[i]
        ax.imshow(matrix[i], cmap="viridis")
        ax.axis("off")
        ax.set_title(f"Slice {i+1}")
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# # Example usage
# depth = 8
# generated_shape = generate_irregular_3d_shape(depth, size=32, irregularity=0.3, growth_chance=0.6)
# visualize_3d_slices(generated_shape, title="Generated Irregular 3D Shape")

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def generate_irregular_3d_shape_with_radius(depth, size, sphere_radius, center_depth, irregularity=0.3, growth_chance=0.6):
    """
    Generate a 3D matrix with a random irregular object constrained within a spherical radius.

    Args:
        depth (int): Number of slices along the z-axis.
        size (int): Dimensions of each 2D slice (size x size).
        sphere_radius (float): Approximate spherical radius of the object.
        center_depth (int): The depth index for the center of the sphere.
        irregularity (float): Controls the randomness in shape generation (0-1).
        growth_chance (float): Probability of growth in neighboring slices (0-1).

    Returns:
        numpy.ndarray: A 3D matrix with the generated shape.
    """
    # Initialize 3D space
    space = np.zeros((depth, size, size), dtype=np.float32)

    # Calculate maximum distance allowed for the sphere
    z, y, x = np.meshgrid(
        np.arange(depth), np.arange(size), np.arange(size), indexing="ij"
    )
    center = (center_depth, size // 2, size // 2)
    distance_from_center = np.sqrt(
        (z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2
    )
    within_radius = distance_from_center <= sphere_radius

    # Start with a random seed at the center slice
    seed_radius = sphere_radius / 3  # Initial seed size is smaller than the sphere
    y_slice, x_slice = np.ogrid[:size, :size]
    seed_mask = (
        (x_slice - center[2]) ** 2 + (y_slice - center[1]) ** 2 <= seed_radius ** 2
    )
    space[center_depth][seed_mask] = 1

    # Grow the object across slices
    for z_idx in range(center_depth - 1, -1, -1):  # Grow upward
        previous_slice = space[z_idx + 1]
        growth_mask = (
            (previous_slice > 0) & (np.random.rand(size, size) < growth_chance)
        )
        space[z_idx] = np.clip(previous_slice + growth_mask, 0, 1)

    for z_idx in range(center_depth + 1, depth):  # Grow downward
        previous_slice = space[z_idx - 1]
        growth_mask = (
            (previous_slice > 0) & (np.random.rand(size, size) < growth_chance)
        )
        space[z_idx] = np.clip(previous_slice + growth_mask, 0, 1)

    # Apply the spherical constraint
    space *= within_radius

    # Add irregularity by random erosion/dilation
    for z_idx in range(depth):
        noise = np.random.rand(size, size)
        space[z_idx] = np.where(noise < irregularity, 0, space[z_idx])

    # Smooth the object to create more natural transitions
    space = gaussian_filter(space, sigma=1)

    # Normalize the matrix
    space /= np.max(space)

    return space

# Visualize the generated 3D shape
def visualize_3d_slices(matrix, title="Irregular 3D Object"):
    """
    Visualizes slices of a 3D matrix.
    """
    depth = matrix.shape[0]
    fig, axes = plt.subplots(1, depth, figsize=(depth * 2, 2))
    for i in range(depth):
        ax = axes[i]
        ax.imshow(matrix[i], cmap="viridis")
        ax.axis("off")
        ax.set_title(f"Slice {i+1}")
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# Example usage
depth = 16
size = 32
sphere_radius = 7
center_depth = 8
generated_shape = generate_irregular_3d_shape_with_radius(
    depth, size, sphere_radius, center_depth, irregularity=0.2, growth_chance=0.7
)
visualize_3d_slices(generated_shape, title="Irregular 3D Object with Radius Constraint")
