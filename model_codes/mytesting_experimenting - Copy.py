# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.ndimage import gaussian_filter

# def generate_irregular_3d_shape_with_center(
#     depth, size, sphere_radius, center_coords, irregularity=0.3, growth_chance=0.6
# ):
#     """
#     Generate an irregular 3D object with an approximate radius and center.

#     Args:
#         depth (int): Number of slices along the z-axis.
#         size (int): Dimensions of each 2D slice (size x size).
#         sphere_radius (float): Approximate spherical radius of the object (in cm).
#         center_coords (tuple): Physical coordinates of the center (z, y, x in cm).
#         irregularity (float): Controls the randomness in shape generation (0-1).
#         growth_chance (float): Probability of growth in neighboring slices (0-1).

#     Returns:
#         numpy.ndarray: A 3D matrix with the generated shape.
#     """
#     # Physical dimensions
#     voxel_size_xy = 0.25  # cm per voxel in X and Y
#     voxel_size_z = 0.5  # cm per voxel in Z

#     # Convert physical center to voxel indices
#     z_center_idx = int(center_coords[0] / voxel_size_z)
#     y_center_idx = int((center_coords[1] + 4) / voxel_size_xy)
#     x_center_idx = int((center_coords[2] + 4) / voxel_size_xy)

#     # Create 3D meshgrid for physical coordinates
#     x_range = np.linspace(-4, 4, size)
#     y_range = np.linspace(-4, 4, size)
#     z_range = np.linspace(0, 3.5, depth)
#     z, y, x = np.meshgrid(z_range, y_range, x_range, indexing="ij")

#     # Calculate distance from the center in physical space
#     center = (z_range[z_center_idx], y_range[y_center_idx], x_range[x_center_idx])
#     distance_from_center = np.sqrt(
#         (z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2
#     )

#     # Constrain within the spherical radius
#     within_radius = distance_from_center <= sphere_radius

#     # Initialize 3D space
#     space = np.zeros((depth, size, size), dtype=np.float32)

#     # Start with a random seed at the center
#     seed_radius = sphere_radius / 4  # Initial seed size is smaller than the sphere
#     y_slice, x_slice = np.ogrid[:size, :size]
#     seed_mask = (
#         (x_slice - x_center_idx) ** 2 + (y_slice - y_center_idx) ** 2
#         <= (seed_radius / voxel_size_xy) ** 2
#     )
#     space[z_center_idx][seed_mask] = 1

#     # Grow the object across slices
#     for z_idx in range(z_center_idx - 1, -1, -1):  # Grow upward
#         previous_slice = space[z_idx + 1]
#         growth_mask = (
#             (previous_slice > 0) & (np.random.rand(size, size) < growth_chance)
#         )
#         space[z_idx] = np.clip(previous_slice + growth_mask, 0, 1)

#     for z_idx in range(z_center_idx + 1, depth):  # Grow downward
#         previous_slice = space[z_idx - 1]
#         growth_mask = (
#             (previous_slice > 0) & (np.random.rand(size, size) < growth_chance)
#         )
#         space[z_idx] = np.clip(previous_slice + growth_mask, 0, 1)

#     # Apply the spherical constraint
#     space *= within_radius

#     # Add irregularity by random erosion/dilation
#     for z_idx in range(depth):
#         noise = np.random.rand(size, size)
#         space[z_idx] = np.where(noise < irregularity, 0, space[z_idx])

#     # Smooth the object to create more natural transitions
#     space = gaussian_filter(space, sigma=1)

#     # Normalize the matrix
#     space /= np.max(space)

#     return space

# # Visualize the generated 3D shape
# def visualize_3d_slices(matrix, title="Irregular 3D Object with Approximate Center"):
#     """
#     Visualizes slices of a 3D matrix.
#     """
#     depth = matrix.shape[0]
#     fig, axes = plt.subplots(1, depth, figsize=(depth * 2, 2))
#     for i in range(depth):
#         ax = axes[i]
#         ax.imshow(matrix[i], cmap="viridis")
#         ax.axis("off")
#         ax.set_title(f"Slice {i+1}")
#     plt.suptitle(title, fontsize=16)
#     plt.tight_layout()
#     plt.show()

# # Example usage
# depth = 7
# size = 32
# sphere_radius = .9  # cm
# center_coords = (1.75, 0, 0)  # Approximate center in physical space (z, y, x in cm)
# generated_shape = generate_irregular_3d_shape_with_center(
#     depth, size, sphere_radius, center_coords, irregularity=0.2, growth_chance=0.7
# )
# visualize_3d_slices(
#     generated_shape, title="Irregular 3D Object with Approximate Center"
# )

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.ndimage import gaussian_filter

# def generate_irregular_3d_shape(depth, size=32, sphere_radius=2, center_depth=3, irregularity=0.3, growth_chance=0.6):
#     """
#     Generate a 3D matrix with a random irregular object centered at a specific depth.

#     Args:
#         depth (int): Number of slices along the z-axis.
#         size (int): Dimensions of each 2D slice (size x size).
#         sphere_radius (float): Approximate spherical radius of the object (in cm).
#         center_depth (float): Depth (z-axis in cm) where the shape should be centered.
#         irregularity (float): Controls the randomness in shape generation (0-1).
#         growth_chance (float): Probability of growth in neighboring slices (0-1).

#     Returns:
#         numpy.ndarray: A 3D matrix with the generated shape.
#     """
#     # Physical dimensions
#     voxel_size_xy = 8 / size  # cm per voxel in X and Y
#     voxel_size_z = 3.5 / depth  # cm per voxel in Z

#     # Convert center depth to voxel index
#     center_slice = int(center_depth / voxel_size_z)

#     # Convert sphere radius to voxel units
#     radius_voxels_xy = int(sphere_radius / voxel_size_xy)
#     radius_voxels_z = int(sphere_radius / voxel_size_z)

#     # Initialize 3D space
#     space = np.zeros((depth, size, size), dtype=np.float32)

#     # Start with a random seed in the central slice
#     y, x = np.ogrid[:size, :size]
#     seed_center = (size // 2, size // 2)
#     seed_mask = (x - seed_center[0]) ** 2 + (y - seed_center[1]) ** 2 <= radius_voxels_xy ** 2
#     space[center_slice][seed_mask] = 1

#     # Grow the object across slices
#     for z in range(center_slice - 1, max(center_slice - radius_voxels_z, -1), -1):  # Grow upward
#         previous_slice = space[z + 1]
#         growth_mask = (
#             (previous_slice > 0) & (np.random.rand(size, size) < growth_chance)
#         )
#         space[z] = np.clip(previous_slice + growth_mask, 0, 1)

#     for z in range(center_slice + 1, min(center_slice + radius_voxels_z, depth)):  # Grow downward
#         previous_slice = space[z - 1]
#         growth_mask = (
#             (previous_slice > 0) & (np.random.rand(size, size) < growth_chance)
#         )
#         space[z] = np.clip(previous_slice + growth_mask, 0, 1)

#     # Add irregularity by random erosion/dilation
#     for z in range(depth):
#         noise = np.random.rand(size, size)
#         space[z] = np.where(noise < irregularity, 0, space[z])

#     # Smooth the object to create more natural transitions
#     space = gaussian_filter(space, sigma=1)

#     # Normalize the matrix
#     space /= np.max(space)

#     return space

# # Visualize the generated 3D shape
# def visualize_3d_slices(matrix, title="Irregular 3D Object"):
#     """
#     Visualizes slices of a 3D matrix.
#     """
#     depth = matrix.shape[0]
#     fig, axes = plt.subplots(1, depth, figsize=(depth * 2, 2))
#     for i in range(depth):
#         ax = axes[i]
#         ax.imshow(matrix[i], cmap="viridis")
#         ax.axis("off")
#         ax.set_title(f"Slice {i+1}")
#     plt.suptitle(title, fontsize=16)
#     plt.tight_layout()
#     plt.show()

# # Example usage
# depth = 7
# size = 32
# sphere_radius = .9  # cm
# center_depth = 1.75  # cm (place the center of the irregular shape)
# generated_shape = generate_irregular_3d_shape(
#     depth, size, sphere_radius, center_depth, irregularity=0.2, growth_chance=0.7
# )
# visualize_3d_slices(
#     generated_shape, title=f"Irregular 3D Object (Radius={sphere_radius} cm, Center Depth={center_depth} cm)"
# )


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def generate_irregular_3d_shape(depth, size=32, sphere_radius=2, center_depth=3, irregularity=0.3, growth_chance=0.6):
    voxel_size_xy = 8 / size  # cm per voxel in X and Y
    voxel_size_z = 3.5 / depth  # cm per voxel in Z

    center_slice = int(center_depth / voxel_size_z)
    radius_voxels_xy = int(sphere_radius / voxel_size_xy)
    radius_voxels_z = int(sphere_radius / voxel_size_z)

    space = np.zeros((depth, size, size), dtype=np.float32)

    y, x = np.ogrid[:size, :size]
    seed_center = (size // 2, size // 2)
    seed_mask = (x - seed_center[0]) ** 2 + (y - seed_center[1]) ** 2 <= radius_voxels_xy ** 2
    space[center_slice][seed_mask] = 1

    # Grow the object across slices
    for z in range(center_slice - 1, max(center_slice - radius_voxels_z, -1), -1):  # Grow upward
        previous_slice = space[z + 1]
        growth_mask = (
            (previous_slice > 0) & (np.random.rand(size, size) < growth_chance)
        )
        z_distance_from_center = abs(z - center_slice)
        allowed_growth = (z_distance_from_center <= radius_voxels_z)
        space[z] = np.clip(previous_slice * allowed_growth + growth_mask, 0, 1)

    for z in range(center_slice + 1, min(center_slice + radius_voxels_z, depth)):  # Grow downward
        previous_slice = space[z - 1]
        growth_mask = (
            (previous_slice > 0) & (np.random.rand(size, size) < growth_chance)
        )
        z_distance_from_center = abs(z - center_slice)
        allowed_growth = (z_distance_from_center <= radius_voxels_z)
        space[z] = np.clip(previous_slice * allowed_growth + growth_mask, 0, 1)

    for z in range(depth):
        noise = np.random.rand(size, size)
        space[z] = np.where(noise < irregularity, 0, space[z])

    space = gaussian_filter(space, sigma=1)
    space /= np.max(space)

    return space

def visualize_3d_slices(matrix, title="Irregular 3D Object"):
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
depth = 7
size = 32
sphere_radius = .5  # cm
center_depth = 1.75  # cm
generated_shape = generate_irregular_3d_shape(
    depth, size, sphere_radius, center_depth, irregularity=0.2, growth_chance=0.7
)
visualize_3d_slices(
    generated_shape, title=f"Irregular 3D Object (Radius={sphere_radius} cm, Center Depth={center_depth} cm)"
)
