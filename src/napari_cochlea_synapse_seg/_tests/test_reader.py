import numpy as np
import zarr
from scipy.ndimage import zoom
from napari_cochlea_synapse_seg import napari_get_reader


# tmp_path is a pytest fixture
def test_reader(tmp_path):
    """An example of how you might test your plugin."""
    
    # write some fake data using your supported file format
    my_test_file = str(tmp_path / "myfile.zarr")
    
    img_data = np.random.rand(6, 20, 20)
    label_data = zoom(
        np.random.randint(10, size=(2, 4, 4)),
        (3, 5, 5)
    )
    
    f = zarr.open(my_test_file, 'w')
    f['3d/raw'] = img_data
    f['3d/raw'].attrs['resolution'] = [1, 1, 1]
    f['3d/raw'].attrs['offset'] = [0, 0, 0]
    f['3d/labeled'] = label_data
    f['3d/labeled'].attrs['resolution'] = [1, 1, 1]
    f['3d/labeled'].attrs['offset'] = [0, 0, 0]
    
    # try to read it back in
    reader = napari_get_reader(my_test_file)
    assert callable(reader)
    
    # make sure we're delivering the right format
    layer_data_list = reader(my_test_file)
    assert isinstance(layer_data_list, list) and len(layer_data_list) > 0
    
    img_data_tuple = layer_data_list[0]
    assert isinstance(img_data_tuple, tuple) and len(img_data_tuple) > 0
    label_data_tuple = layer_data_list[1]
    assert isinstance(label_data_tuple, tuple) and len(label_data_tuple) > 0
    
    # make sure it's the same as it started
    np.testing.assert_allclose(img_data, img_data_tuple[0])
    np.testing.assert_allclose(label_data, label_data_tuple[0])
    
def test_get_reader_pass():
    reader = napari_get_reader("fake.file")
    assert reader is None
