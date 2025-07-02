import torch
import numpy as np
import numpy as np
from scipy.interpolate import interp1d

from Models import * ## self defined 
from random_mask import get_random_mask


inverse_model = Inverse_Operator() 
device = 'cpu'

weight = torch.load('weights/weight_g.pth', map_location='cpu')
inverse_model.load_state_dict(weight)

# weight = torch.load('weights/weight_a.pth', map_location='cpu')
# inverse_model2.load_state_dict(weight)


def n_depth_correction(data, n_depth = .5):
    z_original = np.linspace(0.5, 3.5, 7)

    # New z-axis values for interpolation
    z_new = np.arange(n_depth, 3.1+n_depth, 0.5)

    # Create an empty array to hold the interpolated data
    interpolated_data = np.round(np.zeros((len(z_new), 32, 32)), 2) 

    # Loop through each x, y coordinate and interpolate along the z-axis
    for i in range(32):
        for j in range(32):
            # Extract the z-axis data for the current (x, y) point
            z_slice = data[:, i, j]
            
            # Create the interpolation function
            interp_func = interp1d(z_original, z_slice, kind='linear', bounds_error=False, fill_value="extrapolate")
            
            # Interpolate to the new z values
            interpolated_data[:, i, j] = interp_func(z_new)

    return interpolated_data #data_interpolated

def scale_value(value, old_min=0, old_max=0.35, new_min=0, new_max=0.3):
    """Scales a value from range [old_min, old_max] to [new_min, new_max]."""
    return new_min + (value - old_min) * (new_max - new_min) / (old_max - old_min)

def compute_depth(z_radius, depth):
    if z_radius % 50 == 49:
        t_radius = z_radius + 1  # 0.49/0.99/1.49 --> 0.5, 1, 1.5
    elif z_radius == 70:
        t_radius = z_radius + 5
    else: 
        t_radius = z_radius

    #print("tradius", t_radius)
    # Compute depth of the first layer
    n_depth = (depth - t_radius) % 50

    # Ensure n_depth is not too small
    if n_depth < 21:
        n_depth += 50

    # Prevent n_depth from being equal to depth
    if n_depth == depth:
        n_depth -= 50

    return n_depth



def inference(batchmeasured, b_opt, b_tar):
    '''
    inputs: 
        batch_size = 1 
        batchmeasured: measured data shoud have a dimension of (batch_size, 18, 14) , where 9 source and 14 detector complex value is concated to form 18 x 14 measurement 
        batch_mask: dimension (batch_size, 7, 32, 32)
        b_tar: dimension (batch_size, 2)  where, radius = b_tar[0,1].item() , depth = b_tar[0,0].item()
        b_opt: dimension (batch_size, 4) where , mua0, mus0, mua0, mus0 are the values foe each batch 
    output: 
        final_reconstruction : (batch_size, 7, 32, 32)

    '''
    batch_mask = torch.tensor(get_random_mask(b_tar.numpy()), dtype = torch.float32).to(device)
    batchmeasured = batchmeasured.to(device) #.view(-1, 18, 14) 
    b_opt = b_opt.to(device)

    radius = int(round(b_tar[0,1].item(), 2)*100)
    depth = int(round(b_tar[0,0].item(),2)*100)
    mymask = np.zeros((1,7,1,1))

    n_depth = compute_depth(radius, depth)

    if radius*2 <= 100: 
        n_cen = depth - n_depth 
        n = int((n_cen//50))
        mymask[0,n,0,0] = 1 

    elif radius*2 <= 150: 
        n_cen = depth - n_depth-25
        n = int((n_cen//50))
        mymask[0,n:n+2,0,0] = 1 
    else: 
        n_cen = depth - n_depth
        n = int((n_cen//50))
        mymask[0,n-1:n+2,0,0] = 1 

    # print(n_cen)

    # print(n)
    # print("ndepth", n_depth)
    # print(radius)
    # print(depth)
    # print(mymask.flatten())

    n_depth = 0.01* n_depth
    radius = 0.01* radius 
    depth = 0.01*depth 

    pred_reconstruction = scale_value(inverse_model(batch_mask, batchmeasured, b_opt))
    #pred_reconstruction = (pred_reconstruction/pred_reconstruction.max())*scale_value(0.4*pred_reconstruction + 0.4*pred_reconstruction2 + 0.2*pred_reconstruction3, old_max=0.32, new_max=0.22).max()
    corrected_depth = n_depth_correction(pred_reconstruction.detach().squeeze().numpy(), n_depth=n_depth)
    final_reconstruction = np.expand_dims(corrected_depth, axis = 0)*mymask 

    return final_reconstruction

if __name__ == '__main__':
    batchmeasured = torch.randn(1, 18, 14)
    #batch_mask = torch.randn(1, 7, 32, 32)
    b_opt = torch.randn(1, 4)
    b_tar = torch.randn(1, 2)
    b_tar[0,0] = 2.3
    b_tar[0,1] = 0.7
    output = inference(batchmeasured, b_opt, b_tar)

    print(output.shape)