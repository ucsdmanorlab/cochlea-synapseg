"""
This module provides a custom QWidget class (GTWidget) for use with the napari viewer.
It includes various functionalities to display synapse images, edit and create point and
label annotations, interconvert between points and labels, and save data in as .zarr.
"""
from typing import TYPE_CHECKING
from ._widget_utils import _setup_spin, _limitStretch

from qtpy.QtWidgets import QLabel, QCheckBox, QSpinBox, QDoubleSpinBox, QGroupBox, QVBoxLayout, QGridLayout, QPushButton, QWidget, QComboBox
from napari.layers.utils.stack_utils import stack_to_images
import numpy as np
import tifffile

if TYPE_CHECKING:
    import napari
                
class PreProcessWidget(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.init_ui()

    def init_ui(self):
        self.setLayout(QVBoxLayout())
        self.setup_image_box()
        self.setup_points_box()
        self.setup_labels_box()

        self.update_layer_choices()
        self.viewer.layers.events.inserted.connect(self.update_layer_choices)
        self.viewer.layers.events.removed.connect(self.update_layer_choices)

    def setup_image_box(self):
        self.xyres = 1
        self.zres = 1
        self.img_shape = None

        self.active_image = QComboBox()
        _limitStretch(self.active_image)
        
        box2 = QGroupBox('Image tools')

        xyresbox = QDoubleSpinBox()
        zresbox  = QDoubleSpinBox()
        self.split_choice = QComboBox()
        splitbtn = QPushButton('Split channels')
        splitbtn.clicked.connect(self._split_channels); 

        _setup_spin(self, xyresbox,  minval=0, val=self.xyres, step=0.05, attrname='xyres', dec=4, dtype=float)
        _setup_spin(self, zresbox,  minval=0, val=self.zres, step=0.05, attrname='zres', dec=4, dtype=float)
        self.active_image.currentTextChanged.connect(lambda: self._read_res())
        self.active_image.currentTextChanged.connect(lambda: xyresbox.setValue(self.xyres))
        self.active_image.currentTextChanged.connect(lambda: zresbox.setValue(self.zres))
        self.active_image.currentTextChanged.connect(self._update_split_choice)

        image_gbox = QGridLayout()
        image_gbox.addWidget(QLabel('image layer:'), 0, 0) ; image_gbox.addWidget(self.active_image, 0, 1)
        image_gbox.addWidget(QLabel('xy res:'), 1, 0) 
        image_gbox.addWidget(xyresbox, 1, 1)
        image_gbox.addWidget(QLabel('z res:'), 2, 0) 
        image_gbox.addWidget(zresbox, 2, 1)
        image_gbox.addWidget(self.split_choice, 3, 0)
        image_gbox.addWidget(splitbtn, 3, 1)

        box2.setLayout(image_gbox)

        self.layout().addWidget(box2)

    def setup_points_box(self):
        # Point tools box ################################################################
        self.active_points = QComboBox()

        box3 = QGroupBox('Points tools')
        scalepts = QPushButton('real -> pixel units')
        chbox = QSpinBox(); self.nch=1;  
        ch2zbtn = QPushButton('chan -> z convert')
        snapbtn = QPushButton("Snap to max")
        snapbox = QSpinBox(); self.snap_rad = 3
        _setup_spin(self, snapbox,   minval=0, suff=' px', val=self.snap_rad, attrname='snap_rad')        
        
        _setup_spin(self, chbox, minval=1, maxval=10, val=self.nch, attrname='nch')
        ch2zbtn.clicked.connect(self._convert_ch2z)
        scalepts.clicked.connect(self._scale_points)
        snapbtn.clicked.connect(self._snap_to_max)
        
        points_gbox = QGridLayout()
        points_gbox.addWidget(QLabel('points layer:'), 0, 0) ; points_gbox.addWidget(self.active_points, 0, 1)
        points_gbox.addWidget(scalepts, 1, 0, 1, 2)
        points_gbox.addWidget(chbox, 2, 0)
        points_gbox.addWidget(ch2zbtn, 2, 1)
        points_gbox.addWidget(snapbox, 3, 0)
        points_gbox.addWidget(snapbtn, 3, 1)
        box3.setLayout(points_gbox)

        self.layout().addWidget(box3)
    
    def setup_labels_box(self):
        # Labels tools box ################################################################
        box4 = QGroupBox('Labels tools')
        self.active_label = QComboBox()

        self.labcheck = QCheckBox("Make labels editable")
        self.labcheck.setTristate(False); self.labcheck.setCheckState(False)
        self.labcheck.stateChanged.connect(self._set_editable)
        self.active_label.currentTextChanged.connect(self._set_editable)

        labels_gbox = QGridLayout()
        labels_gbox.addWidget(QLabel('labels layer:'), 0, 0) ; labels_gbox.addWidget(self.active_label, 0, 1)
        labels_gbox.addWidget(self.labcheck, 1, 0, 1, 2); 

        box4.setLayout(labels_gbox)        
        self.layout().addWidget(box4)

    def update_layer_choices(self, event=None):
        img_layers = [l.name for l in self.viewer.layers if l.__class__.__name__ == "Image"]
        point_layers = [l.name for l in self.viewer.layers if l.__class__.__name__ == "Points"]

        img_choice = self.active_image.currentText()
        pts_choice = self.active_points.currentText()
        
        for combo in (self.active_image, self.active_points):
            combo.clear()
        
        self.active_image.addItems(img_layers)
        self.active_points.addItems(point_layers)

        if img_choice in img_layers:
            self.active_image.setCurrentText(img_choice)
        if pts_choice in point_layers:
            self.active_points.setCurrentText(pts_choice)

        for layer in self.viewer.layers:
            layer.events.name.connect(self.update_layer_choices)

    def _update_split_choice(self):
        dims = self.img_shape
        if dims is not None:
            self.split_choice.clear()
            self.split_choice.addItem("None")
            for i, d in enumerate(dims):
                self.split_choice.addItem(f"Axis {i}: {d}")
                if d == np.min(dims):
                    default_choice = i+1
            self.split_choice.setCurrentIndex(default_choice)

    def _convert_ch2z(self):
        try:
            pts = self.viewer.layers[self.active_points.currentText()]
        except:
            print("Points layer not defined.")
            return
        
        pts.data[:,0] = pts.data[:,0]/self.nch
        pts.refresh()
    
    def _split_channels(self):
        try:
            img = self.viewer.layers[self.active_image.currentText()]
        except:
            print("Image layer not defined.")
            return
        
        ll = self.viewer.layers
        layer = img
        if self.split_choice.currentText() == "None":
            print("Choose channel dimension.")
            return
        else:
            axis = int(self.split_choice.currentText().split(':')[0].split(' ')[-1])
        remember_path = layer.source.path
        images = stack_to_images(layer, axis=axis)
        for img in images:
            if hasattr(img.source, '__class__'):
                new_source = img.source.__class__(path=remember_path)
                img._source = new_source
        ll.remove(layer)
        ll.extend(images)

    def _scale_points(self):
        try:
            pts = self.viewer.layers[self.active_points.currentText()]
        except:
            print("Points layer not defined.")
            return
        
        pts.data[:,0] = pts.data[:,0]/self.zres
        pts.data[:,1] = pts.data[:,1]/self.xyres
        pts.data[:,2] = pts.data[:,2]/self.xyres
        
        pts.refresh()

    def _read_res(self):
        try:
            img = self.viewer.layers[self.active_image.currentText()]
        except:
            return
        imgpath = img.source.path

        def _read_tiff_voxel_size(file_path):
            """
            Implemented based on information found in https://pypi.org/project/tifffile
            """

            def _xy_voxel_size(tags, key):
                assert key in ['XResolution', 'YResolution']
                if key in tags:
                    num_pixels, units = tags[key].value
                    return units / num_pixels
                # return default
                return 1.

            with tifffile.TiffFile(file_path) as tiff:
                image_metadata = tiff.imagej_metadata
                if image_metadata is not None:
                    z = image_metadata.get('spacing', 1.)
                else:
                    # default voxel size
                    z = 1.

                tags = tiff.pages[0].tags
                # parse X, Y resolution
                y = _xy_voxel_size(tags, 'YResolution')
                x = _xy_voxel_size(tags, 'XResolution')
                # return voxel size
                return [z, y, x]
            
        if imgpath is not None and imgpath.endswith('.tif'):
            [z, y, x] = _read_tiff_voxel_size(imgpath)
            self.xyres = x
            self.zres = z

        self.img_shape = img.data.shape
        # TODO: add functionality for .czi or other formats?
    
    def _snap_to_max(self):
        should_break=False
        try:
            pts = self.viewer.layers[self.active_points.currentText()].data
        except:
            print("Points layer not defined.")
            should_break=True
        try:
            img = self.viewer.layers[self.active_image.currentText()].data
        except:
            print("Image layer not defined.")
            should_break=True
        if self.snap_rad<=0:
            print("Snap radius must be >0.")
            should_break=True
        if should_break:
            print("Snap to max could not complete. Make sure correct image and points layers are selected.")
            return
        
        #blur_sig = [self.blur_sig_z, self.blur_sig_xy, self.blur_sig_xy]
        #img_inv = gaussian_filter(np.min(img) + np.max(img) - img, blur_sig)
        count = 0

        for pos in pts:
            pos = np.round(pos).astype('int')
            zrange, yrange, xrange, rel_pos = self._get_slices(self.snap_rad, self.snap_rad, pos, img.shape)
            pointIntensity = img[zrange, yrange, xrange]

            shift = np.unravel_index(np.argmax(pointIntensity), pointIntensity.shape)
            shift = np.asarray(shift)-self.snap_rad

            pos = (pos + shift).astype('int')
            pts[count] = pos
            count += 1
        self.viewer.layers[self.active_points.currentText()].refresh()

    def _convert_dask(self, layer_name):
        try:
            self.viewer.layers[layer_name].data.chunks
        except:
            # not a dask
            return
        print('converting layer '+layer_name+' to numpy array')
        self.viewer.layers[layer_name].data = self.viewer.layers[layer_name].data.compute()

    def _set_editable(self):
        if self.labcheck.isChecked():
            try:
                label = self.active_label.currentText()
            except:
                return
            self._convert_dask(self.active_label.currentText())
        

