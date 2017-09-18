import numpy as np
from skimage.transform import resize as skresize
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm

def width_normalize(dataset, reshape_sizes, output_size):
    dataset = dataset.astype('float32')
    dataset /= 255
    output_dataset = []
    for this_shape in tqdm(reshape_sizes):
        w = this_shape if this_shape != 0 else dataset.shape[2]
        h = output_size
        new_dataset = []
        for d in tqdm(dataset):
            nd = skresize(d[0], (h, w))
            padding = output_size - w
            if padding > 0:
                left_padding = round(padding/2)
                right_padding = padding - left_padding
                nd = np.pad(nd, ((0,0),(left_padding, right_padding)), mode='constant')
            
            new_dataset.append(nd.reshape(output_size, output_size, 1))
        output_dataset.append(dataset)
    return output_dataset

def elastic_transform(image, alpha=37.0, sigma=5.5, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)