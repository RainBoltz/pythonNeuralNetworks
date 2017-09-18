import numpy as np
from skimage.transform import resize as skresize

def width_normalize(dataset, reshape_sizes, output_size):
        dataset = dataset.astype('float32')
        dataset /= 255
    output_dataset = []
    for this_shape in reshape_sizes:
        w = this_shape if this_shape != 0 else dataset.shape[2]
        h = output_size
        new_dataset = []
        for d in dataset:
            nd = skresize(d, (h, w))
            padding = output_size - w
            if padding > 0:
                left_padding = round(padding/2)
                right_padding = padding - left_padding
                nd = np.pad(nd, ((0,0),(left_padding, right_padding)), mode='constant')
            
            new_dataset.append(nd)
        output_dataset.append(dataset)
    return output_dataset