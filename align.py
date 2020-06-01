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
                       max_steps=100,
                       descent_rate=1,
                       mesh_regulariser_weight=.1,
                       mesh_regulariser_gaussian_sigma=2,
                       sobel_kernel_size=15,
                       error_gaussian_sigma=2,
                       min_error=None,
                       stopping_criterion='flatten'):
    centroid_shift = get_centroid(target) - get_centroid(image)
    curr_distort_idx = np.stack([
        np.ones(image.shape) * centroid_shift[1],
        np.ones(image.shape) * centroid_shift[0],], axis=-1)
    
    conv_filter = np.array([[0,1,0],[1,0,1],[0,1,0]])
    filter_count = convolve2d(np.ones_like(image), conv_filter, mode='same')
    log = []
    for step_i in range(max_steps):
        #TODO write stopping criterion
        curr_distort_img = distort(
            image,
            distort_idx=[curr_distort_idx[...,0], curr_distort_idx[...,1]],
            interpolation=cv2.INTER_LINEAR)
        sobelx = cv2.Sobel(curr_distort_img,cv2.CV_32F,1,0,ksize=sobel_kernel_size)
        sobely = cv2.Sobel(curr_distort_img,cv2.CV_32F,0,1,ksize=sobel_kernel_size)
        sobel_grad = np.stack((sobely, sobelx), axis=-1)
        sobel_grad /= np.linalg.norm(sobel_grad, axis=-1)[:,:,None] + 1e-10
        
        diff_img = gaussian_filter(target, error_gaussian_sigma, mode='constant') - \
                   gaussian_filter(curr_distort_img, error_gaussian_sigma, mode='constant')
        
        error_reduction = diff_img[:,:,None] * sobel_grad
        local_tension = np.stack([convolve2d(
            curr_distort_idx[...,i],
            conv_filter,
            mode='same')/filter_count - curr_distort_idx[...,i] for i in range(2)], axis=-1)
        # TODO implement gaussian based local tension
        curr_distort_idx += descent_rate * (error_reduction + \
                                mesh_regulariser_weight * local_tension)
    
        log += [(curr_distort_img.copy(), diff_img.copy(), curr_distort_idx[...,0].copy(), curr_distort_idx[...,1].copy())]
    return log