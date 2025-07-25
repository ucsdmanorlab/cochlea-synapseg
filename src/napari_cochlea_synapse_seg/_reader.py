"""
Reader functions for:
-.xml files from the FIJI CellCounter plugin
-.csv, .xls, .XLS files containing CenterX, CenterY, CenterZ columns, outputted from Amira
"""
import numpy as np
import pandas as pd
import zarr
import xml.etree.ElementTree as et
import os
# from aicsimageio import AICSImage


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    # if we know we cannot read the file, we immediately return None.
    # otherwise we return the *function* that can read ``path``.
    if isinstance(path, str): 
        if path.endswith(".xml"):
            return cellcounter_reader_function
        elif path.endswith(".csv"):
            return amira_csv_reader_function
        elif path.endswith(".xls") or path.endswith(".XLS"):
            return amira_xls_reader_function
        elif path.endswith(".zarr"):
            if "raw" in zarr.open(path).keys() and "labeled" in zarr.open(path).keys():
                return zarr_reader_function

        #     return zarr_reader_function
    else:
        return None

# def czi_reader_function(file):
#     #nch = 
#     img = AICSImage(file).get_image_dask_data()#.get_image_dask_data("ZYX", C=1).compute()
#     add_kwargs = {}
#     layer_type = "image"  
#     return [(img, add_kwargs, layer_type)]

def amira_xls_reader_function(xlsfile):

    df = pd.read_excel(xlsfile)
    data = np.array(df[['CenterZ', 'CenterY', 'CenterX']])
    
    # optional kwargs for the corresponding viewer.add_* method
    add_kwargs = {"ndim": 3, 
                  "face_color": 'magenta', 
                  "border_color": 'white',
                  "size": 5,
                  "out_of_slice_display": True,
                  "opacity": 0.7,
                  "symbol": 'x'}

    layer_type = "points"  # optional, default is "image"
    return [(data, add_kwargs, layer_type)]

def amira_csv_reader_function(csvfile):
    
    zyxres = [1, 1, 1] #read_tiff_voxel_size(imfi)

    xyz = np.genfromtxt(csvfile, delimiter=",", skip_header=1)[:,-3::]
    x = xyz[:,0]/zyxres[2]
    y = xyz[:,1]/zyxres[1]
    z = xyz[:,2]/zyxres[0]
    
    data = np.stack((z,y,x), axis=1)

    # optional kwargs for the corresponding viewer.add_* method
    add_kwargs = {}

    layer_type = "points"  # optional, default is "image"
    return [(data, add_kwargs, layer_type)]

def cellcounter_reader_function(xmlfile):
    try:
        tree = et.parse(xmlfile)
    except OSError:
        print('Failed to read XML file {}.'.format(xmlfile))
    
    root = tree.getroot()
    img_props = root[0]
    markers = root[1]
    
    x = np.array([int(elem.text) for elem in markers.findall(".//MarkerX")])
    y = np.array([int(elem.text) for elem in markers.findall(".//MarkerY")])
    z = np.array([int(elem.text) for elem in markers.findall(".//MarkerZ")])
    
    data = np.stack((z,y,x), axis=1)

    # optional kwargs for the corresponding viewer.add_* method
    add_kwargs = {"ndim": 3, 
                  "face_color": 'magenta', 
                  "border_color": 'white',
                  "size": 5,
                  "out_of_slice_display": True,
                  "opacity": 0.7,
                  "symbol": 'x'}

    layer_type = "points"  # optional, default is "image"
    return [(data, add_kwargs, layer_type)]

def zarr_reader_function(path):
    # optional kwargs for the corresponding viewer.add_* method
    img_kwargs = {}
    label_kwargs = {}

    zarr_fi = zarr.open(path)
    img = (zarr_fi['raw'], img_kwargs, "image")
    labels = (zarr_fi['labeled'].astype(int)[:], label_kwargs, "labels")

    return [img, labels]
