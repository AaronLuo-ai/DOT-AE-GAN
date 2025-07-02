import numpy as np
import random
import matplotlib.pyplot as plt 

import numpy as np

import concurrent.futures
import time
import os

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

def get_random_mask(b_tar, mask_ref):
    mask = []
    #batch_size = b_tar.shape[0]
    array = np.zeros((7, 32, 32))
    #print(b_tar.dtype)
    for (depth, radius) in b_tar:
        weight = 2 #*1/radius # random.uniform(2, 2.5)
        mask_i =insert_elliptical_sphere(array, depth, a_radius = radius*weight , b_radius = radius*weight, c_radius = radius, mua=1)
        
        mask.append(mask_i)
    return np.array(mask)* np.any(mask_ref != 0, axis = (2,3), keepdims= True) #(mask_ref.max(axis=(2,3), keepdims = True))



# def get_random_mask_parallel(b_tar, mask_ref, num_workers = 4):
#     #tasks = [10**6, 10**7, 10**8, 10**9]
#     results = []
#     #num_workers = 4  # Specify the number of worker processes
    
#     with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
#         #futures = [executor.submit(get_random_mask, task) for task in tasks]

#         futures = []
#         for i in range(len(b_tar)//num_workers):
#             futures.append(executor.submit(get_random_mask, b_tar[i*num_workers:(i+1)*num_workers], mask_ref[i*num_workers:(i+1)*num_workers]))

#         for future in concurrent.futures.as_completed(futures):
#             try:
#                 results.append(future.result())
#             except Exception as exc:
#                 print(f'Generated an exception: {exc}')

#         return np.vstack(results)

def get_random_mask_parallel(b_tar, mask_ref, num_workers=4):
    results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        step = len(b_tar) // num_workers
        
        for i in range(num_workers):
            start_idx = i * step
            end_idx = (i + 1) * step if i != num_workers - 1 else len(b_tar)
            futures.append(executor.submit(get_random_mask, b_tar[start_idx:end_idx], mask_ref[start_idx:end_idx]))
        
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                print(f'Generated an exception: {exc}')

    return np.vstack(results)
    


if __name__ == '__main__':
    # array = np.zeros((7, 32,32)) 
    # weight = random.uniform(1.5, 1.55)
    # ground_truth = insert_elliptical_sphere(array, depth = 2, a_radius = .5*weight, b_radius=0.5*weight, c_radius = .5, mua=1)

    # fig, axs = plt.subplots(1, 7, figsize=(15, 5))
    # # Plot the first set of images in the first row
    # for x in range(7):
    #     axs[ x].imshow(ground_truth[x])
    #     axs[ x].axis('off')  # Hide the axes for a cleaner look
    # # Adjust the layout to prevent overlap
    # plt.tight_layout()

    # # Show the plots
    # plt.show()

    b_tar = np.ones((16, 2)) 
    mask_ref = np.ones((16, 7, 32,32)) 

    out = get_random_mask_parallel(b_tar, mask_ref, 16)
    print(out.shape)


