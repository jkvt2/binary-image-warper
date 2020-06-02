from align import regress_distortion, distort
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

def make_toy_image(
        image_size=[100,80],
        num_shapes=[3,8],
        num_points=[3,6],
        size=[10,30]):
    image = np.zeros((80,100), dtype=np.float32)
    cv2.fillPoly(image, [
        np.array([
            [15,30],
            [32,20],
            [60,37],
            [52,54],
            [22,48]]),
        np.array([
            [82,10],
            [96,15],
            [82,60],
            [70,54]])], 1)
    return image

def make_toy_distortion(
        image,
        distort_idx,
        num_seeds=50,
        seed_size=(5,30),
        seed_mag=2,
        blur_sigma=3,
        random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    imshape = image.shape[:2]
    distortion_mag = [np.zeros(imshape, dtype=np.float32),
                      np.zeros(imshape, dtype=np.float32)]
    for i in range(2):
        for _ in range(num_seeds):
            new = np.zeros(imshape, dtype=np.float32)
            cv2.circle(new,
                tuple(np.random.randint(imshape)),
                np.random.randint(*seed_size),
                np.random.uniform(-seed_mag, seed_mag), -1)
            distortion_mag[i] += new
    distort_idx = [gaussian_filter(i, blur_sigma) + \
                   np.random.normal(scale=j/10) for i,j in zip(distortion_mag, imshape)]
    distort_img = distort(image, distort_idx)
    return distort_img, distort_idx


if __name__ == '__main__':
    #make toy example
    orig_image = make_toy_image()
    target_distort_image, target_distort_idx = make_toy_distortion(
        orig_image,
        distort_idx=None,
        num_seeds=20,
        seed_size=(5,30),
        seed_mag=10,
        blur_sigma=10,
        random_seed=0)
    plt.subplot(1,2,1)
    plt.imshow(orig_image)
    plt.axis('off')
    plt.title('Input')
    plt.subplot(1,2,2)
    plt.imshow(target_distort_image)
    plt.axis('off')
    plt.title('Desired Output')
    
    #regress the distortion
    pred_distort_idx, log = regress_distortion(
        image=orig_image,
        target=target_distort_image,
        max_steps=1000,
        descent_rate=1,
        mesh_regulariser_weight=1,
        mesh_regulariser_gaussian_sigma=3,
        sobel_kernel_size=15,
        error_gaussian_sigma=1,
        return_logs=True)
    
    #checkboard for visualisation
    checkboard = np.zeros(orig_image.shape, dtype=np.float32)
    h,w = checkboard.shape
    for i in range(h//8):
        for j in range(w//8):
            if j%2 == i%2:
                continue
            checkboard[i*8:i*8+8, j*8:j*8+8] = 1
    
    #visualise progress
    if len(log) > 5:
        disp = [(i,log[i]) for i in (np.arange(5)/4 * (len(log) - 1)).astype(int)]
    else:
        disp = list(enumerate(log))
    plt.figure()
    plt.subplot(1 + len(disp),5,1)
    plt.imshow(target_distort_image)
    plt.title('Image', fontsize=10)
    plt.tick_params(
        axis='both',
        which='both',
        left=False,
        right=False,
        bottom=False,
        top=False,
        labelbottom=False,
        labelleft=False)
    plt.ylabel('Target', fontsize=10)
    plt.subplot(1 + len(disp),5,3)
    plt.imshow(distort(checkboard, target_distort_idx))
    plt.axis('off')
    plt.title('Distortion', fontsize=10)
    plt.subplot(1 + len(disp),5,4)
    plt.imshow(target_distort_idx[0])
    plt.axis('off')
    plt.title('Offset (x)', fontsize=10)
    plt.subplot(1 + len(disp),5,5)
    plt.imshow(target_distort_idx[1])
    plt.axis('off')
    plt.title('Offset (y)', fontsize=10)
    for i,(n,(j,k,l,m)) in enumerate(disp):
        plt.subplot(1 + len(disp),5,6+i*5)
        plt.imshow(j)
        plt.tick_params(
            axis='both',
            which='both',
            left=False,
            right=False,
            bottom=False,
            top=False,
            labelbottom=False,
            labelleft=False)
        plt.ylabel('Iter {}'.format(n), fontsize=10)
        plt.subplot(1 + len(disp),5,7+i*5)
        plt.imshow(k)
        plt.axis('off')
        plt.subplot(1 + len(disp),5,8+i*5)
        plt.imshow(distort(checkboard, (l,m)))
        plt.axis('off')
        plt.subplot(1 + len(disp),5,9+i*5)
        plt.imshow(l)
        plt.axis('off')
        plt.subplot(1 + len(disp),5,10+i*5)
        plt.imshow(m)
        plt.axis('off')
        
    #error graph
    plt.figure()
    plt.plot([np.mean(np.square(target_distort_image - i[0])) for i in log])
    plt.xlabel('Iteration')
    plt.ylabel('Error')