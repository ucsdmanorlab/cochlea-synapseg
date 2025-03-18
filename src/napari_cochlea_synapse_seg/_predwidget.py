"""
This module provides a custom QWidget class (PredWidget) for use with the napari viewer.
It includes various functionalities to display synapse images, edit and create point and
label annotations, interconvert between points and labels, and save data in as .zarr.
"""
from typing import TYPE_CHECKING

from qtpy.QtWidgets import QTabWidget, QLabel, QCheckBox, QSpinBox, QDoubleSpinBox, QGroupBox, QHBoxLayout, QVBoxLayout, QGridLayout, QPushButton, QWidget, QFileDialog, QComboBox, QLineEdit, QCompleter
from scipy.ndimage import gaussian_filter, distance_transform_edt, center_of_mass
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops_table
from skimage.segmentation import watershed
from skimage.util import map_array

from ._reader import napari_get_reader
from ._predict import predict
import os
import numpy as np
import zarr
import tifffile

if TYPE_CHECKING:
    import napari
                
class PredWidget(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.init_ui()

    def init_ui(self):
        self.setLayout(QVBoxLayout())
        self.setup_predict_box()
        self.setup_pred2label_box()

    def setup_predict_box(self):

        pred_btn = QPushButton('Predict')
        pred_btn.clicked.connect(self._predict); 
        show_btn = QPushButton('Show')
        show_btn.clicked.connect(self._show_pred)
        prednshow_btn = QPushButton('Predict and Show')
        prednshow_btn.clicked.connect(lambda: [self._predict(), self._show_pred()])

        browse_model_button = QPushButton("\uD83D\uDD0D"); browse_model_button.setToolTip("Find model file")
        browse_image_button = QPushButton("\uD83D\uDD0D"); browse_image_button.setToolTip("Find image .zarr file")
        
        self.model_path_input = QLineEdit(self)
        self.zarr_path_input = QLineEdit(self)
        browse_model_button.clicked.connect(self._browse_for_model)
        browse_image_button.clicked.connect(self._browse_for_zarr_input)
        #self.model_path_input.textChanged.connect(self._update_completer)
        #self.zarr_path_input.textChanged.connect(self._update_completer)
        
        box = QGroupBox('Predict')
        box.setLayout(QVBoxLayout())
        box2 = QGroupBox('Model path'); box2.setLayout(QHBoxLayout())
        box2.layout().addWidget(self.model_path_input)
        box2.layout().addWidget(browse_model_button)
        box3 = QGroupBox('Input .zarr path'); box3.setLayout(QHBoxLayout())
        box3.layout().addWidget(self.zarr_path_input)
        box3.layout().addWidget(browse_image_button)
        
        box.layout().addWidget(box2)
        box.layout().addWidget(box3)
        
        # box.layout().addWidget(QLabel('Input image:'), 1, 0)
        # box.layout().addWidget(self.input_image, 1, 1)
        # box.layout().addWidget(img_refreshbtn, 1, 2)
        box.layout().addWidget(pred_btn)#, 2, 0, 1, 3)
        box.layout().addWidget(show_btn)
        box.layout().addWidget(prednshow_btn)
        self.layout().addWidget(box)

    def setup_pred2label_box(self):
        self.mask_thresh = 0.0
        self.peak_thresh = 0.1
        self.sig_xy = 0.7
        self.sig_z = 0.5
        self.size_filt = 0

        self.active_image = QComboBox()
        img_refreshbtn = QPushButton("\u27F3"); img_refreshbtn.setToolTip("Refresh")
        img_refreshbtn.clicked.connect(lambda: _update_combos(self, self.active_image, 'Image'))
        _update_combos(self, self.active_image, 'Image', set_index=-1) 
        
        box2 = QGroupBox('Labels from prediction')
        
        mask_thresh_box = QDoubleSpinBox()
        peak_thresh_box  = QDoubleSpinBox()
        sig_xy_box = QDoubleSpinBox()
        sig_z_box = QDoubleSpinBox()
        size_filt_box = QSpinBox()

        pred2label_btn = QPushButton('Prediction to labels')
        pred2label_btn.clicked.connect(self._pred2labels); 
        
        _setup_spin(self, mask_thresh_box,  minval=-1, maxval=1, val=self.mask_thresh, step=0.05, attrname='mask_thresh', dec=2, dtype=float)
        _setup_spin(self, peak_thresh_box,  minval=-1, maxval=1, val=self.peak_thresh, step=0.05, attrname='peak_thresh', dec=2, dtype=float)
        _setup_spin(self, sig_xy_box, minval=0, val=self.sig_xy, step=0.05, attrname='sig_xy', dec=2, dtype=float)
        _setup_spin(self, sig_z_box, minval=0, val=self.sig_z, step=0.05, attrname='sig_z', dec=2, dtype=float)
        _setup_spin(self, size_filt_box, minval=0, maxval=1000, val=self.size_filt, step=1, attrname='size_filt', dtype=int)

        p2l_gbox = QGridLayout()
        p2l_gbox.addWidget(QLabel('pred layer:'), 0, 0) ; p2l_gbox.addWidget(self.active_image, 0, 1)
        p2l_gbox.addWidget(img_refreshbtn, 0, 2)
        p2l_gbox.addWidget(QLabel('mask threshold:'), 1, 0) 
        p2l_gbox.addWidget(mask_thresh_box, 1, 1)
        p2l_gbox.addWidget(QLabel('peak threshold:'), 2, 0)
        p2l_gbox.addWidget(peak_thresh_box, 2, 1)
        p2l_gbox.addWidget(QLabel('sigma xy:'), 3, 0)
        p2l_gbox.addWidget(sig_xy_box, 3, 1)
        p2l_gbox.addWidget(QLabel('sigma z:'), 4, 0)
        p2l_gbox.addWidget(sig_z_box, 4, 1)
        p2l_gbox.addWidget(QLabel('size filter:'), 5, 0)
        p2l_gbox.addWidget(size_filt_box, 5, 1)
        p2l_gbox.addWidget(pred2label_btn, 6, 0, 1, 2)

        box2.setLayout(p2l_gbox)

        self.layout().addWidget(box2)
    
    def _predict(self):
        pred = predict(
            self.model_path_input.text(),
            self.zarr_path_input.text(),
            f'raw') 
        
        zarrfi = zarr.open(self.zarr_path_input.text())
        
        zarrfi['pred'] = pred
        zarrfi['pred'].attrs['offset'] = [0,]*3
        zarrfi['pred'].attrs['resolution'] = [1,]*3
    
    def _show_pred(self):
        zarrfi = zarr.open(self.zarr_path_input.text())
        self.viewer.add_image(zarrfi['raw'], name='input')
        self.viewer.add_image(zarrfi['pred'], name='prediction')

    def _browse_for_model(self):
        model_file = QFileDialog.getOpenFileName(self, "Select Model File")
        print(model_file)
        if model_file[0]:
            self.model_path_input.setText(model_file[0])

    def _path_from_raw_source(self):
        raw_path = self.viewer.layers[self.active_image.currentText()].source.path

        if raw_path.rfind('.zarr')>0:
            raw_path = raw_path[0:raw_path.rfind('.zarr')+5]
        
        directory, zarrfi = os.path.split(raw_path)

        zarrfi = os.path.splitext(zarrfi)[0]+'.zarr'

        self.file_path_input.setText(directory)
        self.file_name_input.setText(zarrfi)
        
    def _browse_for_path(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.file_path_input.setText(directory)

    def _browse_for_zarr(self):
        start_from = self.file_path_input.text()
        directory = QFileDialog.getExistingDirectory(self, "Select Directory", start_from)
        if directory:
            directory, zarrfi = os.path.split(directory)
            self.file_path_input.setText(directory)
            self.file_name_input.setText(zarrfi)

    def _browse_for_zarr_input(self):
        directory = QFileDialog.getExistingDirectory(self, "Select .zarr with raw inside")
        if directory:
            self.zarr_path_input.setText(directory)

    def _ensure_zarr_extension(self):
        file_name = self.file_name_input.text()
        if not file_name.endswith(".zarr"):
            self.file_name_input.setText(file_name + ".zarr")
    
    def _update_completer(self):
        directory = self.file_path_input.text()
        if os.path.isdir(directory):
            files_in_dir = os.listdir(directory)
            completer = QCompleter(files_in_dir, self.file_name_input)
            #completer.setCaseSensitivity(Qt.CaseInsensitive)
            self.file_name_input.setCompleter(completer)

    def _save_zarr(self, threeD=True, twoD=False):

        fileName = os.path.join(self.file_path_input.text(), self.file_name_input.text())

        # TODO: add dialog box to warn if overwriting a file
        # TODO: ensure fileName ends with .zarr

        zarrfi = zarr.open(fileName)

        raw = self.viewer.layers[self.active_image.currentText()].data
        labels = self.viewer.layers[self.active_label.currentText()].data

        for (name, data) in (('raw', raw), ('labeled', labels)):
            is_dask=True
            try:
                data.chunks
            except:
                is_dask=False

            if threeD:
                if is_dask:
                    zarrfi[os.path.join('3d', f'{name}')] = data.compute()
                else:
                    zarrfi[os.path.join('3d', f'{name}')] = data
                zarrfi[os.path.join('3d', f'{name}')].attrs['offset'] = [0,]*3
                zarrfi[os.path.join('3d', f'{name}')].attrs['resolution'] = [1,]*3

            if twoD:
                for z in range(data.shape[0]):
                    if is_dask:
                        zarrfi[os.path.join('2d',f'{name}', str(z))] = np.expand_dims(data[z], axis=0).compute()
                    else:
                        zarrfi[os.path.join('2d',f'{name}', str(z))] = np.expand_dims(data[z], axis=0)
                    zarrfi[os.path.join('2d',f'{name}', str(z))].attrs['offset'] = [0,]*2
                    zarrfi[os.path.join('2d',f'{name}', str(z))].attrs['resolution'] = [1,]*2

    def _convert_dask(self, layer_name):
        # check if is dask array
        try:
            self.viewer.layers[layer_name].data.chunks
        except:
            # not a dask
            return
        print('converting layer '+layer_name+' to numpy array')
        self.viewer.layers[layer_name].data = np.array(self.viewer.layers[layer_name].data)

    def _pred2labels(self):
        try:
            img_layer = self.viewer.layers[self.active_image.currentText()]
        except:
            print("Image layer not defined.")
            return
        self._convert_dask(self.active_image.currentText())
        img = img_layer.data

        blur_sig = [self.sig_z, self.sig_xy, self.sig_xy]
        if np.any(blur_sig):
            dist_map = gaussian_filter(img, blur_sig)
        else:
            dist_map = img
        
        coords = peak_local_max(
                    dist_map,
                    footprint=np.ones((3, 3, 3)),
                    threshold_abs=self.peak_thresh,
                    min_distance=2,
                    )
        mask = np.zeros(dist_map.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers = label(mask)
        mask = img>(self.mask_thresh)
        markers = markers*mask
        segmentation = watershed(-dist_map, markers, mask=mask)

        if self.size_filt>1:
            seg_props = regionprops_table(segmentation, properties=('label', 'num_pixels'))
            in_labels = seg_props['label']

            out_labels = in_labels
            out_labels[seg_props['num_pixels']<self.size_filt] = 0
            segmentation_filt = map_array(segmentation, in_labels, out_labels)
            
            self.viewer.add_labels(segmentation_filt, name='labels from '+self.active_image.currentText()+', '+str(self.size_filt))
        else:
            self.viewer.add_labels(segmentation, name='labels from '+self.active_image.currentText())
    
def _setup_spin(curr_class, spinbox, minval=None, maxval=None, suff=None, val=None, step=None, dec=None, attrname=None, dtype=int):
        if minval is not None:
            spinbox.setMinimum(minval)
        if maxval is not None:
            spinbox.setMaximum(maxval)
        if suff is not None:
            spinbox.setSuffix(suff)
        if val is not None:
            spinbox.setValue(val)
        if step is not None:
            spinbox.setSingleStep(step)
        if dec is not None:
            spinbox.setDecimals(dec)
        if attrname is not None:
            spinbox.valueChanged[dtype].connect(lambda value: _update_attr(curr_class, value, attrname))
            setattr(curr_class, attrname, spinbox.value())

def _update_attr(curr_class, value, attrname):
        print('setting attribute', value, attrname)
        setattr(curr_class, attrname, value)

def _update_combos(curr_class,
        combobox,
        layer_type='Image',
        set_index=None):

        rememberID = combobox.currentIndex()
        combobox.clear(); count = -1
        combolist = []
        for item in curr_class.viewer.layers:
            if layer_type in str(type(item)):
                combobox.addItem(item.name)
                combolist.append(item.name)
                count += 1

        if set_index is not None and (set_index < count or abs(set_index) <= count):
            combobox.setCurrentIndex(combolist.index(combolist[set_index])) #seems redundant but accomodates negative indices
        elif rememberID>=0 and rememberID < count:
            combobox.setCurrentIndex(rememberID)
