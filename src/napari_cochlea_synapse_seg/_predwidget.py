"""
This module provides a custom QWidget class (PredWidget) for use with the napari viewer.
It includes various functionalities to display synapse images, edit and create point and
label annotations, interconvert between points and labels, and save data in as .zarr.
"""
from typing import TYPE_CHECKING
from qtpy.QtWidgets import QLabel, QSpinBox, QDoubleSpinBox, QGroupBox, QHBoxLayout, QVBoxLayout, QGridLayout, QPushButton, QWidget, QFileDialog, QComboBox, QLineEdit, QCompleter, QCheckBox
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops_table
from skimage.segmentation import watershed
from skimage.util import map_array
from napari.utils import progress
from napari.utils.notifications import show_info, show_error

from ._widget_utils import _setup_spin 
from ._predict import predict
from .utils.post_proc import sdt_to_labels
import os
import numpy as np
import zarr
import requests

if TYPE_CHECKING:
    import napari

# TODO (minor): use napari.utils.progress to show predict progress
 
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
        self.show_pred_check = QCheckBox('Show SDT prediction after predict')
        self.show_pred_check.setChecked(False)
        self.show_label_check = QCheckBox('Show labels after predict')
        self.show_pred_check.setChecked(False)

        # Set default model path to ctbp2_sdt_3d_model in this package
        default_model_path = os.path.join(os.path.dirname(__file__), "ctbp2_sdt_3d_model")
        browse_model_button = QPushButton("\uD83D\uDD0D"); browse_model_button.setToolTip("Find model file")
        self.model_path_input = QLineEdit(self)
        self.model_path_input.setText(default_model_path)
        browse_image_button = QPushButton("\uD83D\uDD0D"); browse_image_button.setToolTip("Find image .zarr file")
        
        #self.model_path_input = QLineEdit(self)
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
        box.layout().addWidget(self.show_pred_check)
        box.layout().addWidget(self.show_label_check)
        self.layout().addWidget(box)

    def setup_pred2label_box(self):
        self.mask_thresh = 0.0
        self.peak_thresh = 0.1
        self.min_distance = 2
        self.sig_xy = 0.7
        self.sig_z = 0.5
        self.size_filt = 0

        self.active_image = QComboBox()
        # self.viewer.layers.events.inserted.connect(lambda: _update_combos(self, self.active_image, 'Image'))
        # self.viewer.layers.events.removed.connect(lambda: _update_combos(self, self.active_image, 'Image'))
        # _update_combos(self, self.active_image, 'Image', set_index=-1) 
        self.update_layer_choices()
        self.viewer.layers.events.inserted.connect(self.update_layer_choices)
        self.viewer.layers.events.removed.connect(self.update_layer_choices)

        box2 = QGroupBox('Labels from prediction')
        
        mask_thresh_box = QDoubleSpinBox()
        show_mask_btn = QPushButton('Show mask')
        peak_thresh_box  = QDoubleSpinBox()
        show_peaks_btn = QPushButton('Show peaks')

        settings_btn = QPushButton('Advanced settings'); settings_btn.setCheckable(True)
        settings_box = QGroupBox('Settings')
        settings_box.setVisible(False)
        settings_btn.toggled.connect(settings_box.setVisible)
        
        sig_xy_box = QDoubleSpinBox()
        sig_z_box = QDoubleSpinBox()
        size_filt_box = QSpinBox()
        min_dist_box = QSpinBox()

        pred2label_btn = QPushButton('Prediction to labels')
        
        _setup_spin(self, mask_thresh_box,  minval=-1, maxval=1, val=self.mask_thresh, step=0.05, attrname='mask_thresh', dec=2, dtype=float)
        _setup_spin(self, peak_thresh_box,  minval=-1, maxval=1, val=self.peak_thresh, step=0.05, attrname='peak_thresh', dec=2, dtype=float)
        _setup_spin(self, sig_xy_box, minval=0, val=self.sig_xy, step=0.05, attrname='sig_xy', dec=2, dtype=float)
        _setup_spin(self, sig_z_box, minval=0, val=self.sig_z, step=0.05, attrname='sig_z', dec=2, dtype=float)
        _setup_spin(self, size_filt_box, minval=0, maxval=1000, val=self.size_filt, step=1, attrname='size_filt', dtype=int)
        _setup_spin(self, min_dist_box, minval=0, maxval=1000, val=self.min_distance, step=1, attrname='min_distance', dtype=int)

        show_mask_btn.clicked.connect(lambda: self.viewer.add_image(self.viewer.layers[self.active_image.currentText()].data>self.mask_thresh, name='mask - '+str(self.mask_thresh)))
        show_peaks_btn.clicked.connect(lambda: self.viewer.add_points((self._get_peaks()), name='peaks - '+str(self.peak_thresh), size=5))
        pred2label_btn.clicked.connect(self._pred2labels); 
        
        settings_box.setLayout(QGridLayout())
        settings_box.layout().addWidget(QLabel('sigma xy:'), 0, 0)
        settings_box.layout().addWidget(sig_xy_box, 0, 1)
        settings_box.layout().addWidget(QLabel('sigma z:'), 1, 0)
        settings_box.layout().addWidget(sig_z_box, 1, 1)
        settings_box.layout().addWidget(QLabel('size filter:'), 2, 0)
        settings_box.layout().addWidget(size_filt_box, 2, 1)
        settings_box.layout().addWidget(QLabel('min distance:'), 3, 0)
        settings_box.layout().addWidget(min_dist_box, 3, 1)

        p2l_gbox = QGridLayout()
        p2l_gbox.addWidget(QLabel('pred layer:'), 0, 0) ; p2l_gbox.addWidget(self.active_image, 0,  1, 1, 2)
        p2l_gbox.addWidget(QLabel('mask threshold:'), 1, 0) 
        p2l_gbox.addWidget(mask_thresh_box, 1, 1)
        p2l_gbox.addWidget(show_mask_btn, 1, 2)
        p2l_gbox.addWidget(QLabel('peak threshold:'), 2, 0)
        p2l_gbox.addWidget(peak_thresh_box, 2, 1)
        p2l_gbox.addWidget(show_peaks_btn, 2, 2)
        p2l_gbox.addWidget(settings_btn, 3, 0, 1, 3)
        p2l_gbox.addWidget(settings_box, 4, 0, 1, 3) #QLabel('sigma xy:'), 3, 0)
        # p2l_gbox.addWidget(sig_xy_box, 3, 1)
        # p2l_gbox.addWidget(QLabel('sigma z:'), 4, 0)
        # p2l_gbox.addWidget(sig_z_box, 4, 1)
        # p2l_gbox.addWidget(QLabel('size filter:'), 5, 0)
        # p2l_gbox.addWidget(size_filt_box, 5, 1)
        p2l_gbox.addWidget(pred2label_btn, 6, 0, 1, 3)

        box2.setLayout(p2l_gbox)

        self.layout().addWidget(box2)

    def _download_model_with_napari_progress(self, url, dest, expected_bytes=None):
        def get_remote_file_size(url):
            response = requests.head(url, allow_redirects=True)
            response.raise_for_status()
            return int(response.headers.get("content-length", 0))
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        
        with open(dest, 'wb') as f, progress(total=total, desc="Downloading model") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        expected_bytes = get_remote_file_size(url) if expected_bytes is None else expected_bytes

        if expected_bytes and os.path.getsize(dest) != expected_bytes:
            os.remove(dest)
            raise IOError("Downloaded model file is incomplete. Try again.")

    def _predict(self):
        if self.model_path_input.text() == '':
            show_info("Model path is empty. Please select a model file.")
            return
        elif self.model_path_input.text().endswith('ctbp2_sdt_3d_model'):
            if not os.path.exists(self.model_path_input.text()):
                show_info("Downloading default model...")
                URL = "https://github.com/ucsdmanorlab/cochlea-synapseg/releases/download/v0.1.1/ctbp2_sdt_3d_model"
                self._download_model_with_napari_progress(URL, self.model_path_input.text(), expected_bytes=540535054)
                # with requests.get(URL, stream=True) as r:
                #     r.raise_for_status()
                #     with open(self.model_path_input.text(), 'wb') as f:
                #         for chunk in r.iter_content(chunk_size=8192):
                #             f.write(chunk)

        with progress(desc="Running prediction...(see terminal for progress)"):
            pred = predict(
                self.model_path_input.text(),
                self.zarr_path_input.text(),
                f'raw') 

        zarrfi = zarr.open(self.zarr_path_input.text())
        zarr_res = zarrfi['raw'].attrs.get('resolution', [1,]*3)
        zarrfi['pred'] = pred
        zarrfi['pred'].attrs['offset'] = [0,]*3
        zarrfi['pred'].attrs['resolution'] = zarr_res

        show_info("Prediction complete.")

        labels = sdt_to_labels(pred, 
                  peak_thresh=self.peak_thresh,
                  mask_thresh=self.mask_thresh,
                  strict_peak_thresh=True,
                  blur_sig=[self.sig_z, self.sig_xy, self.sig_xy],
                  size_filt=self.size_filt)
        zarrfi['pred_labels'] = labels
        zarrfi['pred_labels'].attrs['offset'] = [0,]*3
        zarrfi['pred_labels'].attrs['resolution'] = zarr_res

        if self.show_pred_check.isChecked() or self.show_label_check.isChecked():
            zarrfi = zarr.open(self.zarr_path_input.text())
            self.viewer.add_image(zarrfi['raw'], name='input', contrast_limits=[0, 65535])
        if self.show_pred_check.isChecked():
            zarrfi = zarr.open(self.zarr_path_input.text())
            self.viewer.add_image(zarrfi['pred'], name='SDT prediction', contrast_limits=[0,1])
        if self.show_label_check.isChecked():
            self.viewer.add_labels(labels, name='labels from prediction')
    
    def _show_pred(self):
        zarrfi = zarr.open(self.zarr_path_input.text())
        self.viewer.add_image(zarrfi['raw'], name='input', contrast_limits=[0, 65535])
        self.viewer.add_image(zarrfi['pred'], name='prediction', contrast_limits=[0,1])

    def _browse_for_model(self):
        model_file = QFileDialog.getOpenFileName(self, "Select Model File")
        if model_file[0]:
            self.model_path_input.setText(model_file[0])

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


    def _convert_dask(self, layer_name):
        # check if is dask array
        try:
            self.viewer.layers[layer_name].data.chunks
        except:
            # not a dask
            return
        show_info('converting layer '+layer_name+' to numpy array')
        self.viewer.layers[layer_name].data = np.array(self.viewer.layers[layer_name].data)
    
    def _get_peaks(self):
        try:
            img_layer = self.viewer.layers[self.active_image.currentText()]
        except:
            show_error("Prediction layer not defined. Peaks not found.")
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
                    min_distance=self.min_distance,
                    )
        return coords
    def _pred2labels(self):
        try:
            img_layer = self.viewer.layers[self.active_image.currentText()]
        except:
            show_error("Prediction layer not defined. Labels cannot be calculated.")
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
                    min_distance=self.min_distance,
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
    def update_layer_choices(self, event=None):
        img_layers = [l.name for l in self.viewer.layers if l.__class__.__name__ == "Image"]

        img_choice = self.active_image.currentText()
        self.active_image.clear()

        self.active_image.addItems(img_layers)
        if img_choice in img_layers:
            self.active_image.setCurrentText(img_choice)
        elif img_choice == '' and len(img_layers) > 0:
            self.active_image.setCurrentText(img_layers[-1]) # default to most recent layer
        for layer in self.viewer.layers:
            layer.events.name.connect(self.update_layer_choices)
