import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import os
import glob
from sklearn.model_selection import train_test_split

########################################
#   Visualize the GT and masks   #
########################################

depth = [0.5, 1.0, 1.5, 2.0, 2.5, 3, 3.5]

import torch


def show_tensor_image2(img, depth=None):
    img = img.detach().cpu().numpy()
    img = np.squeeze(img)   # remove singleton dims

    # img can now be [7,H,W] or [H,W]
    if img.ndim == 2:
        img = img[None, ...]   # promote to [1,H,W]

    num_slices = img.shape[0]
    slice_positions = np.arange(num_slices)

    # Create subplots
    fig, axes = plt.subplots(1, num_slices, figsize=(15, 5))

    # Plot slices
    for i, pos in enumerate(slice_positions):
        slice_data = img[pos, :, :]

        im = axes[i].imshow(slice_data, vmin=0, vmax=0.32, cmap='jet')

        if depth is not None and len(depth) == num_slices:
            axes[i].set_title(f"Depth {depth[pos]} cm")
        else:
            axes[i].set_title(f"Slice {pos}")

        axes[i].axis('off')

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
    cbar.set_label('Intensity')
    plt.show()

########################################
#   Loading Irregular shapes Dataset   #
########################################

base_path = r"C:\Users\aaron.l\PycharmProjects\DOT-AE-GAN\SIMULATION_to_PERTURBATION\output"

all_X_lesion, all_y_lesion, all_mask_lesion, all_optics_lesion, all_target_lesion = [], [], [], [], []

# Find all lesion folders
lesion_pattern = os.path.join(base_path, "lesion_*")
lesion_folders = glob.glob(lesion_pattern)

for folder_path in lesion_folders:
    folder_name = os.path.basename(folder_path)
    if folder_name.startswith("lesion_"):
        lesion_num = folder_name.split("_")[1]
        try:
            X = np.load(os.path.join(folder_path, f"All_measured_data_phan_lesion{lesion_num}.npy"))
            y = np.load(os.path.join(folder_path, f"All_ground_truth_phan_lesion{lesion_num}.npy"))
            mask = np.load(os.path.join(folder_path, f"All_fine_meshes_phan_lesion{lesion_num}.npy"))
            optics = np.load(os.path.join(folder_path, f"All_background_optics_phan_lesion{lesion_num}.npy"))
            target = np.load(os.path.join(folder_path, f"Target_depth_radius_phan_lesion{lesion_num}.npy"))
            all_X_lesion.append(X)
            all_y_lesion.append(y)
            all_mask_lesion.append(mask)
            all_optics_lesion.append(optics)
            all_target_lesion.append(target)
        except Exception as e:
            print(f"❌ Error loading lesion {lesion_num} at {folder_path}: {e}")

X_irregular = np.concatenate(all_X_lesion, axis=0)
y_irregular = np.concatenate(all_y_lesion, axis=0)
Mask_irregular = np.concatenate(all_mask_lesion, axis=0)
optics_irregular = np.concatenate(all_optics_lesion, axis=0)
target_irregular = np.concatenate(all_target_lesion, axis=0)

# print(f"Final X shape: {X_irregular.shape}")
# print(f"Final y shape: {y_irregular.shape}")
# print(f"Final Mask shape: {Mask_irregular.shape}")
# print(f"Final optics shape: {optics_irregular.shape}")
# print(f"Final target shape: {target_irregular.shape}")


######################################
#   Loading Regular shapes Dataset   #
######################################

X10 = np.load("Data3/all_rad_mua8/All_measured_data_phan.npy")
y10 = np.load("Data3/all_rad_mua8/All_ground_truth_phan.npy") #.transpose((0, 3,1,2))
mask10 = np.load("Data3/all_rad_mua8/All_fine_meshes_phan.npy")
optics10 = np.load("Data3/all_rad_mua8/All_background_optics_phan.npy")
target10 = np.load("Data3/all_rad_mua8/Target_depth_radius_phan.npy")


X11 = np.load("Data3/rad.75/All_measured_data_phan.npy")
y11 = np.load("Data3/rad.75/All_ground_truth_phan.npy") #.transpose((0, 3,1,2))
mask11 = np.load("Data3/rad.75/All_fine_meshes_phan.npy")
optics11 = np.load("Data3/rad.75/All_background_optics_phan.npy")
target11 = np.load("Data3/rad.75/Target_depth_radius_phan.npy")

X12 = np.load("Data3/rad.9/All_measured_data_phan.npy")
y12 = np.load("Data3/rad.9/All_ground_truth_phan.npy") #.transpose((0, 3,1,2))
mask12 = np.load("Data3/rad.9/All_fine_meshes_phan.npy")
optics12 = np.load("Data3/rad.9/All_background_optics_phan.npy")
target12 = np.load("Data3/rad.9/Target_depth_radius_phan.npy")

X13 = np.load("Data3/rad1.05/All_measured_data_phan.npy")
y13 = np.load("Data3/rad1.05/All_ground_truth_phan.npy") #.transpose((0, 3,1,2))
mask13 = np.load("Data3/rad1.05/All_fine_meshes_phan.npy")
optics13 = np.load("Data3/rad1.05/All_background_optics_phan.npy")
target13 = np.load("Data3/rad1.05/Target_depth_radius_phan.npy")

X14 = np.load("Data3/rad1.2/All_measured_data_phan.npy")
y14 = np.load("Data3/rad1.2/All_ground_truth_phan.npy") #.transpose((0, 3,1,2))
mask14 = np.load("Data3/rad1.2/All_fine_meshes_phan.npy")
optics14 = np.load("Data3/rad1.2/All_background_optics_phan.npy")
target14 = np.load("Data3/rad1.2/Target_depth_radius_phan.npy")


X15 = np.load("Data3/depth3.5/All_measured_data_phan.npy")
y15 = np.load("Data3/depth3.5/All_ground_truth_phan.npy") #.transpose((0, 3,1,2))
mask15 = np.load("Data3/depth3.5/All_fine_meshes_phan.npy")
optics15 = np.load("Data3/depth3.5/All_background_optics_phan.npy")
target15 = np.load("Data3/depth3.5/Target_depth_radius_phan.npy")


X16 = np.load("Data3/depthmrad.75/All_measured_data_phan.npy")
y16 = np.load("Data3/depthmrad.75/All_ground_truth_phan.npy") #.transpose((0, 3,1,2))
mask16 = np.load("Data3/depthmrad.75/All_fine_meshes_phan.npy")
optics16 = np.load("Data3/depthmrad.75/All_background_optics_phan.npy")
target16 = np.load("Data3/depthmrad.75/Target_depth_radius_phan.npy")

X17 = np.load("Data3/depthmrad.9/All_measured_data_phan.npy")
y17 = np.load("Data3/depthmrad.9/All_ground_truth_phan.npy") #.transpose((0, 3,1,2))
mask17 = np.load("Data3/depthmrad.9/All_fine_meshes_phan.npy")
optics17 = np.load("Data3/depthmrad.9/All_background_optics_phan.npy")
target17 = np.load("Data3/depthmrad.9/Target_depth_radius_phan.npy")

X18 = np.load("Data3/depthmrad1.05/All_measured_data_phan.npy")
y18 = np.load("Data3/depthmrad1.05/All_ground_truth_phan.npy") #.transpose((0, 3,1,2))
mask18 = np.load("Data3/depthmrad1.05/All_fine_meshes_phan.npy")
optics18 = np.load("Data3/depthmrad1.05/All_background_optics_phan.npy")
target18 = np.load("Data3/depthmrad1.05/Target_depth_radius_phan.npy")

X19 = np.load("Data3/depthmrad1.2/All_measured_data_phan.npy")
y19 = np.load("Data3/depthmrad1.2/All_ground_truth_phan.npy") #.transpose((0, 3,1,2))
mask19 = np.load("Data3/depthmrad1.2/All_fine_meshes_phan.npy")
optics19 = np.load("Data3/depthmrad1.2/All_background_optics_phan.npy")
target19 = np.load("Data3/depthmrad1.2/Target_depth_radius_phan.npy")


X20 = np.load("Data3/depthmrad.5/All_measured_data_phan.npy")
y20 = np.load("Data3/depthmrad.5/All_ground_truth_phan.npy") #.transpose((0, 3,1,2))
mask20 = np.load("Data3/depthmrad.5/All_fine_meshes_phan.npy")
optics20 = np.load("Data3/depthmrad.5/All_background_optics_phan.npy")
target20 = np.load("Data3/depthmrad.5/Target_depth_radius_phan.npy")


X21 = np.load("Data3/rad.5/All_measured_data_phan.npy")
y21 = np.load("Data3/rad.5/All_ground_truth_phan.npy") #.transpose((0, 3,1,2))
mask21 = np.load("Data3/rad.5/All_fine_meshes_phan.npy")
optics21 = np.load("Data3/rad.5/All_background_optics_phan.npy")
target21 = np.load("Data3/rad.5/Target_depth_radius_phan.npy")

X_spherical = np.concatenate([X10, X11, X12, X13, X14, X15, X16, X17, X18, X19, X20, X21], axis=0)
y_spherical = np.concatenate([y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21], axis=0)
Mask_spherical = np.concatenate([mask10, mask11, mask12, mask13, mask14, mask15, mask16, mask17, mask18, mask19, mask20, mask21], axis=0)
optics_spherical = np.concatenate([optics10, optics11, optics12, optics13, optics14, optics15, optics16, optics17, optics18, optics19, optics20, optics21], axis=0)
target_spherical = np.concatenate([target10, target11, target12, target13, target14, target15, target16, target17, target18, target19, target20, target21], axis=0)



##########################################
#          Combine both Dataset          #
##########################################

X = np.concatenate([X_spherical, X_irregular], axis=0)
y = np.concatenate([y_spherical, y_irregular], axis=0)
Mask = np.concatenate([Mask_spherical, Mask_irregular], axis=0)
optics = np.concatenate([optics_spherical, optics_irregular], axis=0)
target = np.concatenate([target_spherical, target_irregular], axis=0)

##############################################
#      3:1 Split on Spherical lesions        #
##############################################
(X_spherical_train, X_spherical_test,
 y_spherical_train, y_spherical_test,
 Mask_spherical_train, Mask_spherical_test,
 optics_spherical_train, optics_spherical_test,
 target_spherical_train, target_spherical_test) = train_test_split(
    X_spherical, y_spherical, Mask_spherical, optics_spherical, target_spherical,
    test_size=0.25, random_state=42
)

##############################################
#      3:1 Split on Irregular lesions        #
##############################################
(X_irregular_train, X_irregular_test,
 y_irregular_train, y_irregular_test,
 Mask_irregular_train, Mask_irregular_test,
 optics_irregular_train, optics_irregular_test,
 target_irregular_train, target_irregular_test) = train_test_split(
    X_irregular, y_irregular, Mask_irregular, optics_irregular, target_irregular,
    test_size=0.25, random_state=42
)

##########################################
#            Training Dataset            #
##########################################
X_train = np.concatenate([X_spherical_train, X_irregular_train], axis=0)
y_train = np.concatenate([y_spherical_train, y_irregular_train], axis=0)
Mask_train = np.concatenate([Mask_spherical_train, Mask_irregular_train], axis=0)
optics_train = np.concatenate([optics_spherical_train, optics_irregular_train], axis=0)
target_train = np.concatenate([target_spherical_train, target_irregular_train], axis=0)

##########################################
#            Testing Dataset             #
##########################################
X_test = np.concatenate([X_irregular_test, X_spherical_test], axis=0)
y_test = np.concatenate([y_irregular_test, y_spherical_test], axis=0)
Mask_test = np.concatenate([Mask_irregular_test, Mask_spherical_test], axis=0)
optics_test = np.concatenate([optics_irregular_test, optics_spherical_test], axis=0)
target_test = np.concatenate([target_irregular_test, target_spherical_test], axis=0)


##########################################
#           Lightening Dataset           #
##########################################

# Placeholder filter kernels (you must define these)
filter_kernel = np.array([[0.05, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]], dtype=np.float32)
filter_kernel2 = np.array([[0.05, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]], dtype=np.float32)

# Assuming you've already loaded and split your data into these variables:
# X_train, y_train, Mask_train, optics_train, tar_train (Combined train)
# X_test, y_test, Mask_test, optics_test, tar_test (Combined test)
# X_irregular, y_irregular, Mask_irregular, optics_irregular, target_irregular (Irregular-only)

class MyDataset(Dataset):
    def __init__(self, X, y, Mask, optics, target, train=True):
        self.X = X
        self.y = y
        self.Mask = Mask
        self.optics = optics
        self.target = target
        self.train = train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_data = self.X[idx]
        mask_data = self.Mask[idx]
        optics_data = self.optics[idx]
        target_data = self.target[idx]
        y_data = self.y[idx]
        # print("shape of x_data:", x_data.shape)
        # print("shape of mask_data:", mask_data.shape)
        # print("shape of optics_data:", optics_data.shape)
        # print("shape of target_data:", target_data.shape)
        # print("shape of y_data:", y_data.shape)

        # Initialize y_final with y_data
        y_final = y_data

        # Apply convolution only if it's the training set and the condition is met
        if self.train and target_data[1] > 0.4:
            convolved_slices = []
            for slice_idx in range(y_data.shape[0]):
                temp_2d = y_data[slice_idx]
                y_convolved = convolve(temp_2d, filter_kernel, mode='nearest')
                y_convolved = y_convolved + temp_2d
                y_convolved = convolve(y_convolved, filter_kernel2, mode='nearest')

                max_val = np.max(y_convolved)
                y_convolved = (y_convolved / (1e-7 + max_val)) * np.max(temp_2d)

                convolved_slices.append(y_convolved)

            y_final = np.stack(convolved_slices, axis=0)

        # Apply the shared transformation for both training and validation
        # Add a channel dimension and duplicate the data to match the model's expected 4 channels

        return (torch.Tensor(x_data),
                torch.Tensor(mask_data),
                torch.Tensor(optics_data),
                torch.Tensor(target_data),
                torch.Tensor(y_final))


def main():
    print(torch.cuda.memory_summary())

    # Optionally, specify a device and enable abbreviated output
    print(torch.cuda.memory_summary(device=torch.device('cuda:0'), abbreviated=True))
    BATCH_SIZE = 16

    train_dataset = MyDataset(
        X_train, y_train, Mask_train, optics_train, target_train, train=True
    )

    test_dataset = MyDataset(
        X_test, y_test, Mask_test, optics_test, target_test, train=False
    )

    # Create the training DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Store all batches in a list
    all_batches = []
    for batch in test_dataloader:
        all_batches.append(batch)

    # Get the second-to-last batch using list indexing
    # We use [-2] to get the second-to-last element
    if len(all_batches) >= 2:
        second_to_last_batch = all_batches[-2]
    else:
        print("There are not enough batches to get the second-to-last one.")
        return
    first_batch = all_batches[1]

    (x_data, mask_data, optics_data, target_data, y_data) = first_batch
    print("x_data shape: ", x_data.shape)
    print("mask_data shape: ", mask_data.shape)
    print("optics_data shape: ", optics_data.shape)
    print("target_data shape: ", target_data.shape)
    print("y_data shape: ", y_data.shape)
    for i in range(min(5, y_data.shape[0])):
        mask = mask_data[i]
        image = y_data[i]
        print("i = ", i)
        print("shape", image.shape)
        print("mask", mask.shape)
        show_tensor_image2(image)
        show_tensor_image2(mask)

if __name__ == '__main__':
    main()