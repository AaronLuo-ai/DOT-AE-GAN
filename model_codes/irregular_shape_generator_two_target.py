import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.draw import ellipsoid
from tqdm import tqdm 
import random 


# Define arrays
depths =  np.linspace(0.25, 3.25, 7)[1:]
radii = np.linspace(0.75,1.2,4) #np.concatenate([np.array([0.5], np.linspace(0.75,1.2,4))]) #np.linspace(0.5,  1.2, 4)
muas = np.linspace(0.12, 0.3, 13)

# bttom bound and top bound 
ua0 = [0.0175, 0.09]
us0 = [3.5, 8.5]
# ua0 = random.uniform(0.065, 0.095) 
# us0 = random.uniform(4.5, 10)



import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.draw import ellipsoid

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.draw import ellipsoid

def generate_two_irregular_shapes(mean_radius_cm, center_depth_cm, volume_shape=(7, 32, 32), voxel_spacing=(0.5, 0.25, 0.25)):
    """
    Generates a 3D volume with two irregular blobs that try not to touch each other.
    """
    # def create_irregular_blob(mean_radius_voxels):
    #     shape = tuple(2 * r + 1 for r in mean_radius_voxels)
    #     ellip = 1 - ellipsoid(*mean_radius_voxels, levelset=True)
    #     noise = np.random.normal(loc=0.0, scale=0.3, size=ellip.shape)
    #     blob = (ellip + noise) > 0
    #     blob = gaussian_filter(blob.astype(float), sigma=0.3)
    #     return blob
    
    def create_irregular_blob(mean_radius_voxels, elongation_factor_range=(1., 1.1)):
        """
        Creates an irregular, possibly elongated 3D blob.
        """
        # Randomly elongate along x, y, z
        elongation = np.random.uniform(*elongation_factor_range, size=3)
        elongated_radius = tuple(int(r * e) for r, e in zip(mean_radius_voxels, elongation))

        # Create base ellipsoid
        shape = tuple(2 * r + 1 for r in elongated_radius)
        ellip = 1 - ellipsoid(*elongated_radius, levelset=True)

        # Add noise
        noise = np.random.normal(loc=0.0, scale=0.7, size=ellip.shape)
        blob = (ellip + noise) > 0

        # Anisotropic smoothing (different sigma for each axis)
        sigma = [0.5 / e for e in elongation]  # smaller sigma for elongated directions
        blob = gaussian_filter(blob.astype(float), sigma=sigma)

        return blob

    def insert_blob(volume, blob, center):
        blob_shape = blob.shape
        blob_center = tuple(s // 2 for s in blob_shape)
        for z in range(blob_shape[0]):
            for x in range(blob_shape[1]):
                for y in range(blob_shape[2]):
                    vz = center[0] - blob_center[0] + z
                    vx = center[1] - blob_center[1] + x
                    vy = center[2] - blob_center[2] + y
                    if 0 <= vz < volume.shape[0] and 0 <= vx < volume.shape[1] and 0 <= vy < volume.shape[2]:
                        volume[vz, vx, vy] += blob[z, x, y]

    # Convert cm to voxel units
    mean_radius_voxels = tuple(int((mean_radius_cm - 0.2*mean_radius_cm)/ sp) for sp in voxel_spacing) # remove 20% of the radius to avoid edge

    # Z-center voxel
    z_cm_range = np.linspace(0.5, 3.5, volume_shape[0])
    z1_idx = np.argmin(np.abs(z_cm_range - center_depth_cm))

    avoid_edge = 6
    margin_x = mean_radius_voxels[1] + avoid_edge
    margin_y = mean_radius_voxels[2] + avoid_edge

    # First center random
    x_center1 = np.random.randint(margin_x, volume_shape[1] - margin_x)
    y_center1 = np.random.randint(margin_y, volume_shape[2] - margin_y)

    # Precompute first center in cm
    x1_cm = (x_center1 - volume_shape[1] // 2) * voxel_spacing[1]
    y1_cm = (y_center1 - volume_shape[2] // 2) * voxel_spacing[2]

    # Try finding second center
    max_attempts = 100
    best_distance = -np.inf
    best_center = (x_center1, y_center1)  # fallback: overlap if necessary

    for attempt in range(max_attempts):
        x_center2 = np.random.randint(margin_x, volume_shape[1] - margin_x)
        y_center2 = np.random.randint(margin_y, volume_shape[2] - margin_y)

        x2_cm = (x_center2 - volume_shape[1] // 2) * voxel_spacing[1]
        y2_cm = (y_center2 - volume_shape[2] // 2) * voxel_spacing[2]

        distance_cm = np.sqrt((x1_cm - x2_cm) ** 2 + (y1_cm - y2_cm) ** 2)

        if distance_cm > best_distance:
            best_distance = distance_cm
            best_center = (x_center2, y_center2)

        # Minimum separation: 2 * radius + small margin (0.5 cm)
        if distance_cm > (2 * mean_radius_cm + 0.5):
            break

    # Use best center found
    x_center2, y_center2 = best_center

    # Ideal radius calculation
    dist1 = np.sqrt(x1_cm ** 2 + y1_cm ** 2)
    x2_cm = (x_center2 - volume_shape[1] // 2) * voxel_spacing[1]
    y2_cm = (y_center2 - volume_shape[2] // 2) * voxel_spacing[2]
    dist2 = np.sqrt(x2_cm ** 2 + y2_cm ** 2)
    ideal_radius = max(dist1, dist2) + mean_radius_cm

    lesion_volume = np.zeros(volume_shape, dtype=np.float32)

    # Generate and insert lesions
    blob1 = create_irregular_blob(mean_radius_voxels)
    insert_blob(lesion_volume, blob1, (z1_idx, x_center1, y_center1))

    blob2 = create_irregular_blob(mean_radius_voxels)
    insert_blob(lesion_volume, blob2, (z1_idx, x_center2, y_center2))

    lesion_volume = gaussian_filter(lesion_volume.astype(float), sigma=0.9)

    return lesion_volume, ideal_radius


def generate_single_irregular_shapes(mean_radius_cm, center_depth_cm, volume_shape=(7, 32, 32), voxel_spacing=(0.5, 0.25, 0.25)):
    """
    Generates a 3D volume with two irregular blobs mimicking breast lesions.
    
    Parameters:
        mean_radius_cm (float): Approximate mean radius of the lesions in cm.
        center_depth_cm (float): z-depth in cm for the first lesion; second will be slightly offset.
        volume_shape (tuple): The shape of the output volume (z, x, y).
        voxel_spacing (tuple): The size of each voxel in cm (z, x, y).
    
    Returns:
        lesion_volume (np.ndarray): 3D volume with two lesion masks combined (float values).
    """
    # def create_irregular_blob(mean_radius_voxels):
    #     # Create base ellipsoid
    #     shape = tuple(2 * r + 1 for r in mean_radius_voxels)
    #     ellip = 1 - ellipsoid(*mean_radius_voxels, levelset=True)
    #     noise = np.random.normal(loc=0.0, scale=0.3, size=ellip.shape)
    #     blob = (ellip + noise) > 0
    #     blob = gaussian_filter(blob.astype(float), sigma=0.3)
    #     return blob
    
    def create_irregular_blob(mean_radius_voxels, elongation_factor_range=(1., 1.1)):
        """
        Creates an irregular, possibly elongated 3D blob.
        """
        # Randomly elongate along x, y, z
        elongation = np.random.uniform(*elongation_factor_range, size=3)
        elongated_radius = tuple(int(r * e) for r, e in zip(mean_radius_voxels, elongation))

        # Create base ellipsoid
        shape = tuple(2 * r + 1 for r in elongated_radius)
        ellip = 1 - ellipsoid(*elongated_radius, levelset=True)

        # Add noise
        noise = np.random.normal(loc=0.0, scale=0.7, size=ellip.shape)
        blob = (ellip + noise) > 0

        # Anisotropic smoothing (different sigma for each axis)
        sigma = [0.5 / e for e in elongation]  # smaller sigma for elongated directions
        blob = gaussian_filter(blob.astype(float), sigma=sigma)

        return blob

    def insert_blob(volume, blob, center):
        blob_shape = blob.shape
        blob_center = tuple(s // 2 for s in blob_shape)

        for z in range(blob_shape[0]):
            for x in range(blob_shape[1]):
                for y in range(blob_shape[2]):
                    vz = center[0] - blob_center[0] + z
                    vx = center[1] - blob_center[1] + x
                    vy = center[2] - blob_center[2] + y

                    if 0 <= vz < volume.shape[0] and 0 <= vx < volume.shape[1] and 0 <= vy < volume.shape[2]:
                        volume[vz, vx, vy] += blob[z, x, y]  # Accumulate in case of overlap

    # Convert cm to voxel units
    mean_radius_voxels = tuple(int((mean_radius_cm- 0.2*mean_radius_cm) / sp) for sp in voxel_spacing) # remove 20% of the radius to avoid edge

    # Z-centers in cm mapped to voxel index
    z_cm_range = np.linspace(0.5, 3.5, volume_shape[0])
    z1_idx = np.argmin(np.abs(z_cm_range - center_depth_cm))
    #z2_idx = np.clip(z1_idx + 2, 0, volume_shape[0]-1)  # Offset 2 slices deeper

    # # X-Y centers offset for spatial separation
    # x_center1, y_center1 = volume_shape[1] // 3, volume_shape[2] // 3
    # x_center2, y_center2 = 2 * volume_shape[1] // 3, 2 * volume_shape[2] // 3

    # avoid_edge = int(2 + (1/mean_radius_cm)*3)
    # # Random X-Y centers within safe bounds
    # margin_x = mean_radius_voxels[1] + avoid_edge # small buffer to avoid edge
    # margin_y = mean_radius_voxels[2] + avoid_edge

    x_center1 = volume_shape[1] // 2
    y_center1 = volume_shape[2] // 2


    lesion_volume = np.zeros(volume_shape, dtype=np.float32)

    # Generate and insert first lesion
    blob1 = create_irregular_blob(mean_radius_voxels)
    insert_blob(lesion_volume, blob1, (z1_idx, x_center1, y_center1))


    lesion_volume = gaussian_filter(lesion_volume.astype(float), sigma=0.9)
    
    return lesion_volume




def insert_elliptical_sphere(array, depth, a_radius, b_radius, c_radius, mua=1):
    z_range = np.linspace(0.5, 3.5, array.shape[0])
    x_range = np.linspace(-4, 4, array.shape[1])
    y_range = np.linspace(-4, 4, array.shape[2])
    
    # Get the index of the depth in the z-axis
    z_index = np.abs(z_range - depth).argmin()
    
    # Get the center index of the array
    center_x = array.shape[1] // 2
    center_y = array.shape[2] // 2
    center_z = z_index
    
    for i in range(array.shape[1]):
        for j in range(array.shape[2]):
            for k in range(array.shape[0]):
                # Calculate distance from the center, considering the ellipsoid
                #distance_test = ((x_range[i] - 0) ** 2 / (a_radius ** 2)) + ((y_range[j] - 0) ** 2 / (b_radius ** 2))
                distance_test = (z_range[k] - depth) ** 2 / (c_radius ** 2)

                distance = ((x_range[i] - 0) ** 2 / (a_radius ** 2)) + ((y_range[j] - 0) ** 2 / (b_radius ** 2)) + ((z_range[k] - depth) ** 2 / (c_radius ** 2))
                if distance <= 1 and distance_test<0.75:
                    array[k, i, j] = mua  # or any other value you choose for the ellipsoid
                
    return array


def insert_cylinder(array, center_depth, radius_xy, height_z, mua=1):
    """
    Insert a cylinder-shaped mask into the 3D array.
    
    Parameters:
        array (np.ndarray): The 3D volume to modify.
        center_depth (float): Center of the cylinder along the z-axis (in cm).
        radius_xy (float): Radius of the cylinder in XY plane (in cm).
        height_z (float): Total height (extent along Z) of the cylinder (in cm).
        mua (float): Value to assign inside the cylinder region.
        
    Returns:
        array (np.ndarray): Modified volume with cylinder mask inserted.
    """
    z_range = np.linspace(0.5, 3.5, array.shape[0])   # Z values in cm
    x_range = np.linspace(-4, 4, array.shape[1])       # X values in cm
    y_range = np.linspace(-4, 4, array.shape[2])       # Y values in cm

    # Define cylinder bounds in Z
    half_height = height_z / 2.0

    for k in range(array.shape[0]):
        z_pos = z_range[k]
        if np.abs(z_pos - center_depth) <= half_height:
            for i in range(array.shape[1]):
                for j in range(array.shape[2]):
                    x_pos = x_range[i]
                    y_pos = y_range[j]
                    # Check if (x, y) is within circle radius
                    if (x_pos**2 + y_pos**2) <= radius_xy**2:
                        array[k, i, j] = mua
                        
    return array


def generate_3d_shape(random_depth, random_radius, random_mua, single_target=False):
    if not single_target:
        Mua_GT_generated, radius_between_two_centers = generate_two_irregular_shapes(mean_radius_cm = random_radius, center_depth_cm = random_depth, volume_shape=(7, 32, 32), voxel_spacing=(0.5, 0.25, 0.25))
        array= np.zeros((7, 32,32)) 
        #print("radius_between_two_centers", radius_between_two_centers)
        mask_gen = insert_elliptical_sphere(array, random_depth, radius_between_two_centers ,radius_between_two_centers, random_radius, mua=1)
        #mask_gen = insert_cylinder(array, random_depth, radius_between_two_centers*1.1, random_radius*2, mua=1)
    else:
        Mua_GT_generated = generate_single_irregular_shapes(mean_radius_cm = random_radius, center_depth_cm = random_depth, volume_shape=(7, 32, 32), voxel_spacing=(0.5, 0.25, 0.25))
        array= np.zeros((7, 32,32)) 
        mask_gen = insert_elliptical_sphere(array, random_depth, random_radius*2 ,random_radius*2, random_radius, mua=1)
        #mask_gen = insert_cylinder(array, random_depth, random_radius*2, random_radius*2, mua=1)
    Mua_GT_generated = (Mua_GT_generated/(Mua_GT_generated.max()+1e-7)) * random_mua
    Mua_GT_generated = np.any(mask_gen, axis=(1,2), keepdims= True) *Mua_GT_generated
    return Mua_GT_generated, mask_gen 


def Generate_Reconstruction_Image2(batch, single_target=False):
    Mua_GT_generateds = []
    mask_gens = []
    optics_gens = []
    
    for b in range(batch):
        random_depth = np.random.choice(depths)
        random_radius = np.random.choice(radii)
        random_mua = np.random.choice(muas)

        Mua_GT_generated, mask_gen = generate_3d_shape(random_depth, random_radius, random_mua, single_target=single_target)

        mask_gens.append(mask_gen)
        Mua_GT_generateds.append(Mua_GT_generated)

        
        ch = random.choice([0, 1, 2])

        ua = random.uniform(ua0[0], ua0[1])
        us = random.uniform(us0[0], us0[1])

        # scaling the optics value on the basis of the input ground truth reconstruction 
        scale = (1+(random_mua/.3)*.02)

        if ch ==0: 
            ua_ = ua*random.uniform(1.001,1.02)*scale
            us_ = us*random.uniform(0.98,0.9999)/scale
        elif ch == 1: 
            ua_ =  ua*random.uniform(0.98,0.9999)/scale
            us_ = us*random.uniform(1.001,1.02)*scale
        else: 
            ua_ = ua*random.uniform(1.001,1.02)*scale 
            us_ =  us*random.uniform(1.001,1.02)*scale 

        optics_gen = [ua, us, ua_, us_]
        optics_gens.append(optics_gen)

    return np.stack(Mua_GT_generateds),  np.stack(mask_gens), np.stack(optics_gens)


def make_data(num = 500000, single_target=False): 
    space_with_objects = []
    masks = []
    opticses = []
    for i in tqdm(range(num)):
        space_with_object, mask, optics = Generate_Reconstruction_Image2(16, single_target=single_target)
        space_with_objects.append(space_with_object)
        masks.append(mask)
        opticses.append(optics)
    space_with_objects = np.concatenate(space_with_objects, axis = 0)
    masks = np.concatenate(masks, axis = 0)
    opticses = np.concatenate(opticses, axis = 0)

    if single_target:
        with open('Data4/objects_random2.npy', 'wb') as f: 
            np.save(f , space_with_objects)
        with open('Data4/masks_random2.npy', 'wb') as f: 
            np.save(f , masks)
        with open('Data4/optics_random2.npy', 'wb') as f: 
            np.save(f , opticses)
    else:
        with open('Data4/objects_random3.npy', 'wb') as f: 
            np.save(f , space_with_objects)
        with open('Data4/masks_random3.npy', 'wb') as f: 
            np.save(f , masks)
        with open('Data4/optics_random3.npy', 'wb') as f: 
            np.save(f , opticses)
    print("Data saved")
    return  opticses.shape


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    #lesion_volume = generate_two_irregular_shapes(mean_radius_cm=.9, center_depth_cm=2.0)
    #lesion_volume, _ =  generate_3d_shape(random_depth = 2.0, random_radius = 1.2, random_mua = .3)

    single_target = True

    def downsample_area_numpy(x):
        # x: (batch, 7, 32, 32)
        batch, channels, h, w = x.shape
        assert h % 8 == 0 and w % 8 == 0, "Input size must be divisible by 8"
        
        # Reshape to average over 4x4 blocks
        x = x.reshape(batch, channels, 8, h // 8, 8, w // 8)
        x_down = x.mean(axis=(3, 5))  # average over 4x4 blocks
        return x_down

    if single_target: 
        lesion_volume, mask, optics = Generate_Reconstruction_Image2(5, single_target=True)
        #lesion_max = lesion_volume.max()
        #lesion_volume = downsample_area_numpy(lesion_volume)>0.05* lesion_max
        print("Single Target")

        print(optics)
        for i in range(lesion_volume.shape[1]):
            plt.subplot(2,1,1)
            plt.imshow(lesion_volume[0][i], cmap='gray')
            plt.subplot(2,1,2)
            plt.imshow(mask[0][i], cmap='gray')
            plt.title(f"Z-slice {i}")
            plt.axis('off')
            plt.show()
    else:
        lesion_volume, mask, optics = Generate_Reconstruction_Image2(5, single_target=False)
        lesion_max = lesion_volume.max()
        lesion_volume = downsample_area_numpy(lesion_volume)>0.05* lesion_max
        print("Two Target")

        print(optics)
        for i in range(lesion_volume.shape[1]):
            plt.subplot(2,1,1)
            plt.imshow(lesion_volume[0][i], cmap='gray')
            plt.subplot(2,1,2)
            plt.imshow(mask[0][i], cmap='gray')
            plt.title(f"Z-slice {i}")
            plt.axis('off')
            plt.show()

    #make_data(num = 5000, single_target= True)
    #make_data(num = 5000, single_target= False)
    make_data(num = 5000, single_target= True)