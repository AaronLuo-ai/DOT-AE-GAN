import numpy as np
import random
import matplotlib.pyplot as plt 

import numpy as np


def insert_cylinder(array, center_depth, radius_xy, height_z = 100, mua=1):
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


def insert_elliptical_sphere(array, depth, a_radius, b_radius, c_radius, mua=1):
    #print(a_radius)

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



def get_random_mask(b_tar, mask_ref = 0):
    mask = []
    #batch_size = b_tar.shape[0]
    
    #print(b_tar.dtype)
    for (depth, radius) in b_tar:
        array1 = np.zeros((7, 32, 32))
        array2 = np.zeros((7, 32, 32))
        if radius <1.2:
            weight = 2 # random.uniform(2, 2.5)
        else:
            weight = 1
            radius = min(radius, 3.7) # 3.7 is the max radius

        mask_i = insert_elliptical_sphere(array1, depth, a_radius = radius*weight , b_radius = radius*weight, c_radius = radius, mua=1)
        mask_c = insert_cylinder(array2, depth, radius_xy = radius*weight , height_z=  100, mua=1)
        mask_i = mask_c * mask_i.any(axis = (-2,-1), keepdims = True) #* np.any(mask_ref != 0, axis = (2,3), keepdims= True) #(mask_ref.max(axis=(2,3), keepdims = True))
        mask.append(mask_i)
    return np.array(mask) #$* np.any(mask_ref != 0, axis = (2,3), keepdims= True) #(mask_ref.max(axis=(2,3), keepdims = True))