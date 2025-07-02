import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.path as mpltPath
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import random
from tqdm import tqdm 

# Define arrays
depths =  np.linspace(0, 3.5, 8)[1:]
radii = np.linspace(0.75,  1.2, 4)
muas = np.linspace(0.12, 0.3, 13)

# bttom bound and top bound 
ua0 = [0.0175, 0.09]
us0 = [3.5, 8.5]
# ua0 = random.uniform(0.065, 0.095) 
# us0 = random.uniform(4.5, 10)


def generate_filled_object2(num_points1, num_points2, array_size=(100, 100), smoothing_factor=0.5):
    # Generate random points in 2D
    ch = random.choice([1,2])
    points = np.random.rand(num_points1, 2)
    points2 = np.random.rand(6, 2)


    #print(points2.shape)
    if ch == 1: 
        points = points*0.9
        points2 = points2*0.9 + 0.1

    else: 
        points[:,0:1] = points[:,0:1]*0.9+  0.1
        points[:,1:2] = points[:,1:2]*0.9 

        points2[:,0:1] = points2[:,0:1]*0.9
        points2[:,1:2] = points2[:,1:2]*0.9 + 0.1
    
    # Compute convex hull
    hull = ConvexHull(points)
    hull2 = ConvexHull(points2)

    
    # Create grid points
    x, y = np.meshgrid(np.linspace(0, 1, array_size[0]), np.linspace(0, 1, array_size[1]))
    grid_points = np.column_stack((x.ravel(), y.ravel()))
    
    # Determine which grid points are inside the convex hull
    filled_mask = np.zeros(array_size, dtype=bool)
    
    # Create a Path object from the hull vertices
    hull_path = mpltPath.Path(hull.points[hull.vertices])
    hull_path2 =  mpltPath.Path(hull2.points[hull2.vertices])
    
    # Check each grid point if it's inside the convex hull
    for i in range(array_size[0]):
        for j in range(array_size[1]):
            if hull_path.contains_point([x[i, j], y[i, j]]) or hull_path2.contains_point([x[i, j], y[i, j]]) :
                filled_mask[i, j] = True
    
    return filled_mask



def generate_filled_object(num_points, array_size=(100, 100), smoothing_factor=0.5):
    # Generate random points in 2D
    points = np.random.rand(num_points, 2)
    
    # Compute convex hull
    hull = ConvexHull(points)
    
    # Create grid points
    x, y = np.meshgrid(np.linspace(0, 1, array_size[0]), np.linspace(0, 1, array_size[1]))
    grid_points = np.column_stack((x.ravel(), y.ravel()))
    
    # Determine which grid points are inside the convex hull
    filled_mask = np.zeros(array_size, dtype=bool)
    
    # Create a Path object from the hull vertices
    hull_path = mpltPath.Path(hull.points[hull.vertices])
    
    # Check each grid point if it's inside the convex hull
    for i in range(array_size[0]):
        for j in range(array_size[1]):
            if hull_path.contains_point([x[i, j], y[i, j]]):
                filled_mask[i, j] = True
    
    return filled_mask


def calculate_circle_radius_at_depth(z_index, depth, radius, z_length, z_shape):
    # Calculate the depth of the current layer
    layer_depth = (z_index + 0.5) * z_length / z_shape
    # Calculate the radius of the circle at the current layer
    if radius**2 - (depth - layer_depth)**2 >0:
        circle_radius = np.sqrt(radius**2 - (depth - layer_depth)**2) * 2
    else: 
        circle_radius = 0
    return circle_radius

def calculate_circle_radii_at_all_depths(depth, radius, z_length, z_shape):
    circle_radii = []
    for z_index in range(z_shape):
        circle_radius = calculate_circle_radius_at_depth(z_index, depth, radius, z_length, z_shape)
        circle_radii.append(circle_radius)
    return circle_radii


filter_sizes = [7]

#filter_size = np.random.choice(filter_sizes)
filter_kernel2 = np.ones((1, 3, 3))
filter_kernel2 /= np.sum(filter_kernel2)


rand_points = [7, 5, 3,8] #7, 5, 7,9]

def generate_3d_shape(depth, radius, mua):
    z_length = 3.5  # cm
    z_shape = 8
    # Calculate circle radii at each depth
    circle_radii = calculate_circle_radii_at_all_depths(depth, radius, z_length, z_shape)

    circle_radii = circle_radii[1:]

    #print(circle_radii)

    space_with_object = []
    for rad in circle_radii: 
        if radius< .75:
            rad = rad - rad*0.5
        else: 
            rad = rad - rad*0.2

        if rad > 0.1: 
            pixel_ratio = np.ceil(rad/4*32).astype(int) # full is 8 . half is 4. which is ok when considered radius 

            filled_object = generate_filled_object2(random.choice(rand_points),random.choice(rand_points))
            filled_object = filled_object.astype(np.uint8) * 255
            new_shape = (pixel_ratio, pixel_ratio)
            #print("newshape:", new_shape)
            resized_array = cv2.resize(filled_object.astype(np.uint8) * 255, new_shape, interpolation=cv2.INTER_LINEAR)

            row_pad = (32 - resized_array.shape[0]) // 2
            col_pad = (32 - resized_array.shape[1]) // 2
            slice = np.pad(resized_array, ((row_pad, row_pad), (col_pad, col_pad)), mode='constant')

            slice = cv2.resize(slice.astype(np.uint8) * 255, (32,32), interpolation=cv2.INTER_LINEAR)
            slice = slice>0

            #print(slice.shape) 

            space_with_object.append(slice.astype('uint8'))     
        else: 
            slice = np.zeros((32, 32), dtype = 'uint8')
            #print(slice.dtype)
            space_with_object.append(slice)

    space_with_object = np.stack(space_with_object)
    space_with_object = space_with_object.astype(np.float32)

    
    # Define the filter kernel
    filter_size = np.random.choice(filter_sizes)
    filter_kernel = np.ones((1, filter_size, filter_size))
    filter_kernel /= np.sum(filter_kernel)

    space_with_object_avg = convolve(space_with_object, filter_kernel, mode='nearest')
    space_with_object_avg = convolve(space_with_object_avg, filter_kernel2, mode='nearest')

    space_with_object_avg = space_with_object_avg/(space_with_object_avg.max()+1e-7) #axis = (-2, -1), keepdims=True)+ 1e-7)


    return space_with_object_avg*mua, space_with_object



def insert_sphere(array, depth, radius, mua = 1):
    z_range = np.linspace(0.5, 3.5, array.shape[0])
    x_range = np.linspace(-4, 4, array.shape[1])
    y_range = np.linspace(-4, 4, array.shape[2])
    
    # Get the index of the depth in the z-axis
    #z_index = np.abs(z_range - depth).argmin()
    
    # Get the center index of the array
    # center_x = array.shape[1] // 2
    # center_y = array.shape[2] // 2
    # center_z = z_index
    
    for i in range(array.shape[1]):
        for j in range(array.shape[2]):
            for k in range(array.shape[0]):
                # Calculate distance from the center
                distance = np.sqrt((x_range[i] - 0) ** 2 + (y_range[j] - 0) ** 2 + (z_range[k] - depth) ** 2)
                if distance <= radius:
                    array[k, i, j] = 1.0  # or any other value you choose for the sphere
                
    return array


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




def Generate_Reconstruction_Image(batch):
    Mua_GT_generateds = []
    mask_gens = []
    optics_gens = []
    
    for b in range(batch):
        random_depth = np.random.choice(depths)
        random_radius = np.random.choice(radii)
        random_mua = np.random.choice(muas)

        Mua_GT_generated, mask_gen_prev = generate_3d_shape(random_depth, random_radius, random_mua)

        
        
        array = np.zeros((7, 32,32)) 
        
        #mask_gen_prev = np.any(mask_gen_prev.reshape(7, -1), axis =1, keepdims= False)
        #print(mask_gen_prev.shape)

        # generate eliptical mask keeping the z dim fix and x y random multiple from (1.5 to 2.5) 
        weight =  random.uniform(2, 2.5)
        mask_gen = insert_elliptical_sphere(array, random_depth, random_radius*weight ,random_radius*weight, random_radius, mua=1)
        
        #mask_gen = np.array( [mask_gen_prev[i]*mask_gen[i] for i in range(7)]) # to remove mask wehere there is no ground truth 
        #mask_gen = mask_gen * np.any(mask_gen_prev, axis =(1,2), keepdims= True)
        Mua_GT_generated = np.any(mask_gen, axis=(1,2), keepdims= True) *Mua_GT_generated #* np.any(mask_gen_prev, axis=(1,2), keepdims= True)
        mask_gen = mask_gen* np.any(Mua_GT_generated, axis=(1,2), keepdims= True) 

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


def Generate_Reconstruction_Image2(batch):
    Mua_GT_generateds = []
    mask_gens = []
    optics_gens = []
    mask_gens_not_rand = []
    
    for b in range(batch):
        random_depth = np.random.choice(depths)
        random_radius = np.random.choice(radii)
        random_mua = np.random.choice(muas)

        Mua_GT_generated, mask_gen_prev = generate_3d_shape(random_depth, random_radius, random_mua)

        
        
        array = np.zeros((7, 32,32)) 
        
        #mask_gen_prev = np.any(mask_gen_prev.reshape(7, -1), axis =1, keepdims= False)
        #print(mask_gen_prev.shape)

        # generate eliptical mask keeping the z dim fix and x y random multiple from (1.5 to 2.5) 
        weight =  random.uniform(1.75, 2.25)
        mask_gen = insert_elliptical_sphere(array, random_depth, random_radius*weight ,random_radius*weight, random_radius, mua=1)
        mask_gen_not_rand = insert_elliptical_sphere(array, random_depth, random_radius*2 ,random_radius*2, random_radius, mua=1)
        
        #mask_gen = np.array( [mask_gen_prev[i]*mask_gen[i] for i in range(7)]) # to remove mask wehere there is no ground truth 
        #mask_gen = mask_gen * np.any(mask_gen_prev, axis =(1,2), keepdims= True)
        Mua_GT_generated = np.any(mask_gen, axis=(1,2), keepdims= True) *Mua_GT_generated #* np.any(mask_gen_prev, axis=(1,2), keepdims= True)
        mask_gen = mask_gen* np.any(Mua_GT_generated, axis=(1,2), keepdims= True) 

        mask_gens.append(mask_gen)
        Mua_GT_generateds.append(Mua_GT_generated)
        mask_gens_not_rand.append(mask_gen_not_rand)



        
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

    return np.stack(Mua_GT_generateds),  np.stack(mask_gens), np.stack(optics_gens), np.stack(mask_gens_not_rand)




def make_data(num = 10000000):
    space_with_objects = []
    masks = []
    opticses = []
    masks_not_rand = []
    for i in tqdm(range(num)):
        space_with_object, mask, optics, mask_not_rand = Generate_Reconstruction_Image2(16)
        
        space_with_objects.append(space_with_object)
        masks.append(mask)
        opticses.append(optics)
        masks_not_rand.append(mask_not_rand)


    space_with_objects = np.concatenate(space_with_objects, axis = 0)
    masks = np.concatenate(masks, axis = 0)
    opticses = np.concatenate(opticses, axis = 0)
    masks_not_rand = np.concatenate(masks_not_rand, axis = 0)


    with open('objects_random.npy', 'wb') as f: 
        np.save(f , space_with_objects)
    with open('masks_random.npy', 'wb') as f: 
        np.save(f , masks)
    with open('optics_random.npy', 'wb') as f: 
        np.save(f , opticses)

    with open('masks_not_random.npy', 'wb') as f: 
        np.save(f , masks_not_rand)

    return  opticses.shape

if __name__ == '__main__':
    # depth = 3
    # radius = .5
    # space_with_object, mask, optics = Generate_Reconstruction_Image(4)
    # # filled_object = generate_filled_object(7)

    # # Create a figure and a set of subplots
    # fig, axs = plt.subplots(2, 7, figsize=(15, 5))

    # # Plot the first set of images in the first row
    # for x in range(7):
    #     axs[0, x].imshow(space_with_object[0][x])
    #     axs[0, x].axis('off')  # Hide the axes for a cleaner look

    # axs[0, 0].set_title(f'First Row Title max {space_with_object[0].max()}') #, fontsize=16, loc='left', pad=20)
    # # Plot the second set of images in the second row
    # for x in range(7):
    #     axs[1, x].imshow(mask[0][x])
    #     axs[1, x].axis('off')  # Hide the axes for a cleaner look

    # # Adjust the layout to prevent overlap
    # plt.tight_layout()

    # # Show the plots
    # plt.show()

    x = make_data(10000)
    print(x)
