import numpy as np
import skimage

def random_crop(image, max_margin):
    height, width, n_channels = image.shape
    x_start = np.random.randint(0, int(height * max_margin))
    x_end = np.random.randint(int(height * (1-max_margin)), height)
    y_start = np.random.randint(0, int(width * max_margin))
    y_end = np.random.randint(int(width * (1-max_margin)), width)
    return image[x_start:x_end, y_start:y_end, :]

def random_flip(image, vertical_flip, horizontal_flip):
    if np.random.rand() < horizontal_flip:
        image = image[:,::-1,:]
    if np.random.rand() < vertical_flip:
        iamge = image[::-1,:,:]
    return image

def random_affine(image, max_zoom, max_angle_rad, max_shear, max_shift):
    rand_zoom = np.random.uniform(1-max_zoom, 1+max_zoom, 2)
    rand_angle_rad = np.random.uniform(-max_angle_rad, max_angle_rad)
    rand_shear = np.random.uniform(-max_shear, max_shear)
    height, width, n_channels = image.shape
    rand_shift = [np.random.uniform(-height*max_shift, height*max_shift),
                  np.random.uniform(-width*max_shift, width*max_shift)]
    T = skimage.transform.AffineTransform(scale=rand_zoom,
                                          shear=rand_shear,
                                          rotation=rand_angle_rad,
                                          translation=rand_shift)
    return skimage.transform.warp(image, T.inverse, mode='reflect')

def random_transform(image, shape=(299,299),
                     max_zoom=0.2, max_angle_rad=0.6,
                     max_shear=0.2, max_shift=0.1,
                     vertical_flip=0.3, horizontal_flip=0.3,
                     max_margin=0.1):
    output = random_affine(image, max_zoom, max_angle_rad, max_shear, max_shift)
    output = random_flip(output, vertical_flip, horizontal_flip)
    output = random_crop(output, max_margin)
    return skimage.transform.resize(output, shape)