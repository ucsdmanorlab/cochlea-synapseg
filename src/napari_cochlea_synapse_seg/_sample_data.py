"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/stable/plugins/guides.html?#sample-data

Replace code below according to your needs.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.util import random_noise

def normalize(im, maxval=255, dtype=np.uint8):
    imin = np.min(im)
    imax = np.max(im)
    imout = (im - imin)/(imax - imin)*maxval
    return imout.astype(dtype)

def make_sample_data():
    """Generates an image"""
    # Return list of tuples
    # [(data1, add_image_kwargs1), (data2, add_image_kwargs2)]
    # Check the documentation for more information about the
    # add_image_kwargs
    # https://napari.org/stable/api/napari.Viewer.html#napari.Viewer.add_image

    image_size = (50, 512, 512) #3D
    Nsynapses = 50
    synapse_size = (3, 5, 5) # gaussian radii
    synapse_mag_max = 700
    synapse_mag_min = 300
    
    imout = np.zeros(image_size, dtype=np.float32)
    for i in range(Nsynapses):
        # keep synapses away from edges:
        z = np.random.randint(synapse_size[0], image_size[0]-synapse_size[0])
        y = np.random.randint(synapse_size[1], image_size[1]-synapse_size[1])
        x = np.random.randint(synapse_size[2], image_size[2]-synapse_size[2])

        imout[z, y, x] = np.random.rand(1)*(synapse_mag_max - synapse_mag_min) + synapse_mag_min

    imout = gaussian_filter(imout, sigma=synapse_size)
    
    return [(normalize(imout), {"name": "simulated_synapses"})]

def make_sample_data_with_noise():
    """Generates an image"""
    # Return list of tuples
    # [(data1, add_image_kwargs1), (data2, add_image_kwargs2)]
    # Check the documentation for more information about the
    # add_image_kwargs
    # https://napari.org/stable/api/napari.Viewer.html#napari.Viewer.add_image

    image_size = (50, 512, 512) #3D
    Nsynapses = 50
    synapse_size = (3, 5, 5) # gaussian radii
    synapse_mag_max = 700
    synapse_mag_min = 300
    bk_mag = 0.1
    
    imout = np.ones(image_size, dtype=np.float32)*bk_mag
    for i in range(Nsynapses):
        # keep synapses away from edges:
        z = np.random.randint(synapse_size[0], image_size[0]-synapse_size[0])
        y = np.random.randint(synapse_size[1], image_size[1]-synapse_size[1])
        x = np.random.randint(synapse_size[2], image_size[2]-synapse_size[2])

        imout[z, y, x] = np.random.rand(1)*(synapse_mag_max - synapse_mag_min) + synapse_mag_min

    imout = gaussian_filter(imout, sigma=synapse_size)
    
    imout = random_noise(imout, mode='poisson')
    imout = random_noise(imout, mode='speckle')
    
    return [(normalize(imout), {"name": "simulated_synapses_noise"})]

def make_sample_data_pairs():
    """Generates an image"""
    # Return list of tuples
    # [(data1, add_image_kwargs1), (data2, add_image_kwargs2)]
    # Check the documentation for more information about the
    # add_image_kwargs
    # https://napari.org/stable/api/napari.Viewer.html#napari.Viewer.add_image

    image_size = (50, 512, 512) #3D
    Nsynapses = 50
    synapse_size = (3, 3, 3) # gaussian radii
    synapse_mag_max = 700
    synapse_mag_min = 300
    
    post_syn_prob = 0.9
    post_syn_dist = 5

    pre_syn = np.zeros(image_size, dtype=np.float32)
    post_syn = np.zeros(image_size, dtype=np.float32)

    for i in range(Nsynapses):
        # keep synapses away from edges:
        z = np.random.randint(synapse_size[0], image_size[0]-synapse_size[0])
        y = np.random.randint(synapse_size[1], image_size[1]-synapse_size[1])
        x = np.random.randint(synapse_size[2], image_size[2]-synapse_size[2])

        pre_syn[z, y, x] = np.random.rand(1)*(synapse_mag_max - synapse_mag_min) + synapse_mag_min
        if np.random.rand(1) < post_syn_prob:
            z2 = z + np.random.randint(-post_syn_dist, post_syn_dist)
            y2 = y + np.random.randint(-post_syn_dist, post_syn_dist)
            x2 = x + np.random.randint(-post_syn_dist, post_syn_dist)

            if (z2 >= 0 and z2 < image_size[0] and
                y2 >= 0 and y2 < image_size[1] and
                x2 >= 0 and x2 < image_size[2]):
                post_syn[z2, y2, x2] = np.random.rand(1)*(synapse_mag_max - synapse_mag_min) + synapse_mag_min
    pre_syn = gaussian_filter(pre_syn, sigma=synapse_size)
    post_syn = gaussian_filter(post_syn, sigma=synapse_size)

    return [
        (normalize(pre_syn), {"name": "pre_synaptic", "colormap": "green"}),
        (normalize(post_syn), {"name": "post_synaptic", "colormap": "magenta", "blending": "additive"}),
    ]