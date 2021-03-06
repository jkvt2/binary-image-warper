import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d

def distort(image, distort_idx, interpolation=cv2.INTER_NEAREST):
    orig_idx = np.meshgrid(*map(range, image.shape))
    distort_img = cv2.remap(
        src=image,
        map1=(orig_idx[1].T + distort_idx[1]).astype(np.float32),
        map2=(orig_idx[0].T + distort_idx[0]).astype(np.float32),
        interpolation=interpolation)
    return distort_img

def get_centroid(image):
    return np.argwhere(image).sum(0)/np.sum(image)

def regress_distortion(image, target,
                       num_scales=None,
                       max_steps=100,
                       descent_rate=1,
                       mesh_regulariser_weight=.1,
                       mesh_regulariser_gaussian_sigma=2,
                       sobel_kernel_size=15,
                       error_gaussian_sigma=2,
                       stopping_criterion='flatten',
                       return_logs=False):
    '''
    Regresses a dense mesh to transform *image* into *target*
    
    Parameters:
        image (ndarray): binary 2d image that you want to warp into the target
        target (ndarray): binary 2d image that you want *image* to be warped into
        num_scales (int): number of exponential scales to operate at (0.5x, 0.25x, ...). By default, this is a function of the longer dimension of the image.
        max_steps (int): maximum number of steps that the regressor will run before stopping (default: 100)
        descent_rate (float): (default: 1)
        mesh_regulariser_weight (float): how strongly the predicted mesh should conform to a regular mesh (default: 0.1)
        mesh_regulariser_gaussian_sigma (int): this affects the size of a node's neighborhood for local warp (default:2)
        sobel_kernel_size (int): (default: 15)
        error_gaussian_sigma (int): (default: 2)
        stopping_criterion (str): 
            flatten: stop when the exponential moving average graph has stopped decreasing
            (default: flatten)
        return_logs (bool): whether to return the complete log of each iteration step
    
    Returns:
        distort_idx (tuple): (distort_idx_x, distort_idx_y) the x and y offsets defining the mesh
        logs (list): if return_logs is True, this is returned. This is a list of tuples, with each tuple being \
(distorted image, error signal, distort_idx_x, distort_idx_y,)
    '''
    if num_scales is None:
        num_scales = max(1, int(np.log2(max(image.shape)))-5)
        print('Automatically setting {} scales'.format(num_scales))
    
    imshape = np.array(image.shape)
    centroid_shift = get_centroid(image) - get_centroid(target)
    scale = 2 ** num_scales
    dims = (imshape/scale).astype(np.uint16)
    curr_distort_idx = np.stack([
                np.ones(dims) * centroid_shift[0]/scale,
                np.ones(dims) * centroid_shift[1]/scale,], axis=-1)
    if return_logs:log = []
    for _scale in range(num_scales-1,-1,-1):
        scale = 2 ** _scale
        dims = (imshape/scale).astype(np.uint16)
        if _scale != 0:
            resized_image = cv2.resize(
                image,
                tuple(dims[::-1]))
            resized_target = cv2.resize(
                target,
                tuple(dims[::-1]))
        else:
            resized_image = image
            resized_target = target
        curr_distort_idx = cv2.resize(
                    curr_distort_idx,
                    tuple(dims[::-1]),
                    interpolation=cv2.INTER_NEAREST) * 2
        
        conv_filter = np.array([[0,1,0],[1,0,1],[0,1,0]])
        filter_count = convolve2d(np.ones_like(resized_image), conv_filter, mode='same')
        if stopping_criterion == 'flatten':errors = []
        for step_i in range(max_steps):
            curr_distort_img = distort(
                resized_image,
                distort_idx=[curr_distort_idx[...,0], curr_distort_idx[...,1]],
                interpolation=cv2.INTER_LINEAR)
            sobelx = cv2.Sobel(curr_distort_img,cv2.CV_32F,1,0,ksize=sobel_kernel_size)
            sobely = cv2.Sobel(curr_distort_img,cv2.CV_32F,0,1,ksize=sobel_kernel_size)
            sobel_grad = np.stack((sobely, sobelx), axis=-1)
            sobel_grad /= np.linalg.norm(sobel_grad, axis=-1)[:,:,None] + 1e-10
            
            diff_img = gaussian_filter(resized_target, error_gaussian_sigma, mode='constant') - \
                       gaussian_filter(curr_distort_img, error_gaussian_sigma, mode='constant')
            if stopping_criterion == 'flatten':
                this_error = np.mean(np.square(resized_target - curr_distort_img))
                if step_i == 0:
                    last_error = this_error
                errors += [.9 * last_error + .1 * this_error]
                last_error = this_error
            
            error_reduction = diff_img[:,:,None] * sobel_grad
            local_tension = np.stack([convolve2d(
                curr_distort_idx[...,i],
                conv_filter,
                mode='same')/filter_count - curr_distort_idx[...,i] for i in range(2)], axis=-1)
            # TODO implement gaussian based local tension
            curr_distort_idx += descent_rate * (error_reduction + \
                                    mesh_regulariser_weight * local_tension)
            if return_logs:
                print(curr_distort_idx.shape)
                save_curr_distort_idx = cv2.resize(
                    curr_distort_idx,
                    tuple(imshape[::-1]),
                    interpolation=cv2.INTER_NEAREST) * scale
                save_curr_distort_img = cv2.resize(
                    curr_distort_img,
                    tuple(imshape[::-1]),
                    interpolation=cv2.INTER_NEAREST) * scale
                log += [(save_curr_distort_img, diff_img.copy(), save_curr_distort_idx[...,0], save_curr_distort_idx[...,1])]
            if stopping_criterion == 'flatten':
                if step_i > 5:
                    if np.mean(errors[-5:]) < last_error * 1.01:
                        break
    if return_logs:
        return curr_distort_idx, log
    else:
        return curr_distort_idx

print(regress_distortion.__doc__)