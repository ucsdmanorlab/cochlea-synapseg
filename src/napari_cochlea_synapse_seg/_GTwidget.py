"""
This module provides a custom QWidget class (GTWidget) for use with the napari viewer.
It includes various functionalities to display synapse images, edit and create point and
label annotations, interconvert between points and labels, and save data in as .zarr.
"""
from typing import TYPE_CHECKING

from qtpy.QtWidgets import QTabWidget, QLabel, QCheckBox, QSpinBox, QDoubleSpinBox, QGroupBox, QHBoxLayout, QVBoxLayout, QGridLayout, QPushButton, QWidget, QFileDialog, QComboBox, QLineEdit, QCompleter
from scipy.ndimage import gaussian_filter, distance_transform_edt, center_of_mass
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.segmentation import watershed
from napari.layers.utils.stack_utils import stack_to_images

from ._reader import napari_get_reader
import os
import numpy as np
import zarr
import tifffile

if TYPE_CHECKING:
    import napari
                
class GTWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.init_ui()

    def init_ui(self):
        self.setLayout(QVBoxLayout())
        self.setup_image_box()
        self.setup_points_box()
        self.setup_labels_box()
        self.setup_save_box()

        self.update_layer_choices()
        self.viewer.layers.events.inserted.connect(self.update_layer_choices)
        self.viewer.layers.events.removed.connect(self.update_layer_choices)

    def setup_image_box(self):
        self.xyres = 1
        self.zres = 1

        self.active_image = QComboBox()
        # _update_combos(self, self.active_image, 'Image', set_index=-1) 
        
        box2 = QGroupBox('Image tools')

        xyresbox = QDoubleSpinBox()
        zresbox  = QDoubleSpinBox()
        splitbtn = QPushButton('Split channels')
        splitbtn.clicked.connect(self._split_channels); #splitbtn.clicked.connect(lambda: _update_combos(self, self.active_image, 'Image'))

        _setup_spin(self, xyresbox,  minval=0, val=self.xyres, step=0.05, attrname='xyres', dec=4, dtype=float)
        _setup_spin(self, zresbox,  minval=0, val=self.zres, step=0.05, attrname='zres', dec=4, dtype=float)
        self.active_image.currentTextChanged.connect(lambda: self._read_res())
        self.active_image.currentTextChanged.connect(lambda: xyresbox.setValue(self.xyres))
        self.active_image.currentTextChanged.connect(lambda: zresbox.setValue(self.zres))
        
        image_gbox = QGridLayout()
        image_gbox.addWidget(QLabel('image layer:'), 0, 0) ; image_gbox.addWidget(self.active_image, 0, 1)
        image_gbox.addWidget(QLabel('xy res:'), 1, 0) 
        image_gbox.addWidget(xyresbox, 1, 1)
        image_gbox.addWidget(QLabel('z res:'), 2, 0) 
        image_gbox.addWidget(zresbox, 2, 1)
        image_gbox.addWidget(splitbtn, 3, 0, 1, 2)

        box2.setLayout(image_gbox)

        self.layout().addWidget(box2)

        # self.viewer.layers.events.inserted.connect(lambda: _update_combos(self, self.active_image, 'Image'))
        # self.viewer.layers.events.removed.connect(lambda: _update_combos(self, self.active_image, 'Image'))

    def setup_points_box(self):
        # Point tools box ################################################################
        self.active_points = QComboBox()
        # _update_combos(self, self.active_points, 'Points', set_index=-1)
        # self.viewer.layers.events.inserted.connect(lambda: _update_combos(self, self.active_points, 'Points'))
        # self.viewer.layers.events.removed.connect(lambda: _update_combos(self, self.active_points, 'Points'))

        box3 = QGroupBox('Points tools')
        scalepts = QPushButton('real -> pixel units')
        chbox = QSpinBox(); self.nch=1;  
        ch2zbtn = QPushButton('chan -> z convert')
        
        _setup_spin(self, chbox, minval=1, maxval=10, val=self.nch, attrname='nch')
        ch2zbtn.clicked.connect(self._convert_ch2z)
        scalepts.clicked.connect(self._scale_points)

        points_gbox = QGridLayout()
        points_gbox.addWidget(QLabel('points layer:'), 0, 0) ; points_gbox.addWidget(self.active_points, 0, 1)
        points_gbox.addWidget(scalepts, 1, 0, 1, 2)
        points_gbox.addWidget(chbox, 2, 0)
        points_gbox.addWidget(ch2zbtn, 2, 1)
        box3.setLayout(points_gbox)

        # Points2labels box ##############################################################
        box5 = QGroupBox('Points to labels')

        ptsbtn = QPushButton("New points layer")
        rxybtn = QPushButton("Rotate to xy")
        p2mbtn = QPushButton("Auto-adjust z")

        zbox = QSpinBox()
        zbox.setMinimum(-50)
        zbox.setMaximum(50)#self.zmax)

        snapbtn = QPushButton("Snap to max")

        p2lbtn = QPushButton("Points to labels")
        advbtn = QPushButton("Advanced settings"); advbtn.setCheckable(True)
                
        # Advanced settings
        self.rad_xy = 6
        self.rad_z = 4
        self.max_rad_xy = 2
        self.max_rad_z = 2
        self.snap_rad = 2
        self.blur_sig_xy = 0.7
        self.blur_sig_z = 0.5
        self.solidity_thresh = 0.8
        self.threshold = 0.5

        box5b = QGroupBox('Advanced settings')
        radxybox = QSpinBox(); 
        radzbox = QSpinBox(); 
        snapbox = QSpinBox(); 
        mradxybox = QSpinBox() ; 
        mradzbox = QSpinBox(); 
        sigxybox = QDoubleSpinBox(); 
        sigzbox = QDoubleSpinBox(); 
        solidbox = QDoubleSpinBox();
        threshbox = QDoubleSpinBox();
        wshedcombo = QComboBox();
        self.wshed_type = 'Image'

        _setup_spin(self, radxybox,  minval=1, suff=' px', val=self.rad_xy, attrname='rad_xy')
        _setup_spin(self, radzbox,   minval=0, suff=' px', val=self.rad_z, attrname='rad_z')
        _setup_spin(self, snapbox,   minval=0, suff=' px', val=self.snap_rad, attrname='snap_rad')
        _setup_spin(self, mradxybox, minval=0, suff=' px', val=self.max_rad_xy, attrname='max_rad_xy')
        _setup_spin(self, mradzbox,  minval=0, suff=' px', val=self.max_rad_z, attrname='max_rad_z')
        _setup_spin(self, sigxybox,  minval=0, suff=' px', val=self.blur_sig_xy, step=0.1, attrname='blur_sig_xy', dtype=float)
        _setup_spin(self, sigzbox,   minval=0, suff=' px', val=self.blur_sig_z, step=0.1, attrname='blur_sig_z', dtype=float)
        _setup_spin(self, solidbox,  minval=0, maxval=1, val=self.solidity_thresh, step=0.05, attrname='solidity_thresh', dtype=float)
        _setup_spin(self, threshbox,  minval=0, maxval=1, val=self.threshold, step=0.05, attrname='threshold', dtype=float)
        wshedcombo.addItem('Image'); wshedcombo.addItem('Distance')
        wshedcombo.currentTextChanged.connect(lambda name: _update_attr(self, name, 'wshed_type'))

        ptsbtn.clicked.connect(self._new_pts)
        # ptsbtn.clicked.connect(lambda: _update_combos(self, self.active_points,'Points', set_index=-1))
        p2mbtn.clicked.connect(self._auto_z)
        rxybtn.clicked.connect(self._rxy)
        zbox.valueChanged[int].connect(self._change_z)
        zbox.valueChanged[int].connect(lambda: zbox.setValue(0))
        snapbtn.clicked.connect(self._snap_to_max)
        p2lbtn.clicked.connect(self._points2labels)
        # p2lbtn.clicked.connect(lambda: _update_combos(self,self.active_label, 'Labels'))
        # p2lbtn.clicked.connect(lambda: _update_combos(self,self.active_merge_label, 'Labels', set_index=-1))
        
        box5b.setVisible(False)
        advbtn.toggled.connect(box5b.setVisible)
        
        # _update_combos(self, self.active_points, 'Points')

        gbox5b = QGridLayout()
        gbox5b.addWidget(QLabel('threshold xy rad:'), 0, 0); gbox5b.addWidget(radxybox, 0, 1)
        gbox5b.addWidget(QLabel('threshold z rad:'), 1, 0); gbox5b.addWidget(radzbox, 1, 1)
        gbox5b.addWidget(QLabel('snap to max rad:'), 2, 0); gbox5b.addWidget(snapbox, 2, 1)
        gbox5b.addWidget(QLabel('local max xy rad:'), 3, 0); gbox5b.addWidget(mradxybox, 3, 1)
        gbox5b.addWidget(QLabel('local max z rad:'), 4, 0); gbox5b.addWidget(mradzbox, 4, 1)
        gbox5b.addWidget(QLabel('gaussian xy rad:'), 5, 0); gbox5b.addWidget(sigxybox, 5, 1)
        gbox5b.addWidget(QLabel('gaussian z rad:'), 6, 0); gbox5b.addWidget(sigzbox, 6, 1)
        gbox5b.addWidget(QLabel('solidity:'), 7, 0); gbox5b.addWidget(solidbox, 7, 1)
        gbox5b.addWidget(QLabel('threshold:'), 8, 0); gbox5b.addWidget(threshbox, 8, 1)
        gbox5b.addWidget(QLabel('watershed type:'), 9, 0); gbox5b.addWidget(wshedcombo, 9, 1)
        box5b.setLayout(gbox5b)

        p2l_gbox = QGridLayout()
        p2l_gbox.addWidget(ptsbtn, 0, 0, 1, 2)
        p2l_gbox.addWidget(rxybtn, 1, 0)
        p2l_gbox.addWidget(p2mbtn, 1, 1)
        p2l_gbox.addWidget(QLabel('manually edit z:'), 2, 0)
        p2l_gbox.addWidget(zbox, 2, 1)
        p2l_gbox.addWidget(snapbtn, 3, 0, 1, 2)  
        p2l_gbox.addWidget(p2lbtn, 4, 0, 1, 2)  
        p2l_gbox.addWidget(advbtn, 5, 0, 1, 2)
        p2l_gbox.addWidget(box5b, 6, 0, 1, 2)
        
        box5.setLayout(p2l_gbox)
        
        self.layout().addWidget(box3)
        self.layout().addWidget(box5)

    def setup_labels_box(self):
        self.active_label = QComboBox()
    
        # Label tools box ################################################################
        self.labcheck = QCheckBox("Make labels editable")
        self.labcheck.setTristate(False); self.labcheck.setCheckState(False)
        self.labcheck.stateChanged.connect(self._set_editable)
        self.active_label.currentTextChanged.connect(self._set_editable)
        # _update_combos(self, self.active_label, 'Labels', set_index=-1)
        # self.viewer.layers.events.inserted.connect(lambda: _update_combos(self, self.active_label, 'Labels'))
        # self.viewer.layers.events.removed.connect(lambda: _update_combos(self, self.active_label, 'Labels'))

        box4 = QGroupBox('Labels tools')
        self.labelbox = QSpinBox(); self.rem_label = 1
        addbtn = QPushButton("New label")
        addbtn.clicked.connect(self._add_label)
        removebtn = QPushButton("Remove label")
        self.maxlab = QLabel("max label: ")
        self.active_label.currentTextChanged.connect(self._set_max_label)
        self.active_label.currentTextChanged.connect(self._connect_label_events)

        self.active_merge_label = QComboBox(); 
        mlsbtn = QPushButton("Merge labels")        
        l2pbtn = QPushButton("Labels to points")
        selectLabelsbtn = QPushButton("Keep labels from points")
        # self.viewer.layers.events.inserted.connect(lambda: _update_combos(self, self.active_merge_label, 'Labels'))
        # self.viewer.layers.events.removed.connect(lambda: _update_combos(self, self.active_merge_label, 'Labels'))


        mlsbtn.clicked.connect(self._merge_labels)
        # mlsbtn.clicked.connect(lambda: _update_combos(self,self.active_merge_label, 'Labels', set_index=-1))
        mlsbtn.clicked.connect(self._set_max_label)
        removebtn.clicked.connect(self._remove_label)
        l2pbtn.clicked.connect(self._labels2points)
        # l2pbtn.clicked.connect(lambda: _update_combos(self, self.active_points, 'Points'))
        selectLabelsbtn.clicked.connect(self._remove_labels_wo_points)
        _setup_spin(self, self.labelbox, minval=1, maxval=1000, val=self.rem_label, attrname='rem_label')

        labels_gbox = QGridLayout()
        labels_gbox.addWidget(QLabel('labels layer:'), 0, 0) ; labels_gbox.addWidget(self.active_label, 0, 1)
        labels_gbox.addWidget(self.labcheck, 1, 0); labels_gbox.addWidget(addbtn, 1, 1)
        labels_gbox.addWidget(self.labelbox, 2, 0)
        labels_gbox.addWidget(removebtn, 2, 1)
        #labels_gbox.addWidget(self.maxlab, 3, 0)
        labels_gbox.addWidget(QLabel('merge labels:'), 4, 0); 
        labels_gbox.addWidget(self.active_merge_label, 4, 1); 
        labels_gbox.addWidget(mlsbtn, 5, 0, 1, 2)
        labels_gbox.addWidget(l2pbtn, 6, 0, 1, 2)
        labels_gbox.addWidget(selectLabelsbtn, 7, 0, 1, 2)

        box4.setLayout(labels_gbox)

        # _update_combos(self, self.active_label, 'Labels')
        # _update_combos(self, self.active_merge_label,'Labels')
        
        self.layout().addWidget(box4)

    def update_layer_choices(self, event=None):
        img_layers = [l.name for l in self.viewer.layers if l.__class__.__name__ == "Image"]
        label_layers = [l.name for l in self.viewer.layers if l.__class__.__name__ == "Labels"]
        point_layers = [l.name for l in self.viewer.layers if l.__class__.__name__ == "Points"]

        img_choice = self.active_image.currentText()
        pts_choice = self.active_points.currentText()
        label_choice = self.active_label.currentText()
        merge_label_choice = self.active_merge_label.currentText()
        
        for combo in (self.active_image, self.active_points, self.active_label, self.active_merge_label):
            combo.clear()
        
        self.active_image.addItems(img_layers)
        self.active_points.addItems(point_layers)
        self.active_label.addItems(label_layers)
        self.active_merge_label.addItems(label_layers)

        if img_choice in img_layers:
            self.active_image.setCurrentText(img_choice)
        if pts_choice in point_layers:
            self.active_points.setCurrentText(pts_choice)
        if label_choice in label_layers:
            self.active_label.setCurrentText(label_choice)
        if merge_label_choice in label_layers:
            self.active_merge_label.setCurrentText(merge_label_choice)

        for layer in self.viewer.layers:
            layer.events.name.connect(self.update_layer_choices)

    def _add_label(self):
        ''' set selected label to the next available number '''
        lab = self.active_label.currentText()
        if lab == '':
            return
        
        n = self.viewer.layers[lab].data.max()
        self.viewer.layers[lab].selected_label = (n + 1)


    def setup_save_box(self):
        box6 = QGroupBox('Save zarr')
        save3Dbtn = QPushButton("Save zarr")
        #save23Dbtn = QPushButton("Save 2D and 3D")
        browse_dir_button = QPushButton("\uD83D\uDCC1"); browse_dir_button.setToolTip("Browse for directory")
        browse_zarr_button = QPushButton("\uD83D\uDD0D"); browse_zarr_button.setToolTip("Find existing zarr")
        from_source_button = QPushButton("from source"); from_source_button.setToolTip("Select location from raw source");
        
        self.file_path_input = QLineEdit(self)
        self.file_name_input = QLineEdit(self)

        browse_dir_button.clicked.connect(self._browse_for_path)
        browse_zarr_button.clicked.connect(self._browse_for_zarr)
        self.file_name_input.editingFinished.connect(self._ensure_zarr_extension)
        self.file_path_input.textChanged.connect(self._update_completer)
        save3Dbtn.clicked.connect(lambda: self._save_zarr(threeD=False, twoD=False))
        #save23Dbtn.clicked.connect(lambda: self._save_zarr(threeD=True, twoD=True))
        from_source_button.clicked.connect(self._path_from_raw_source)

        box6.setLayout(QVBoxLayout())
        box6a = QGroupBox('File path'); box6a.setLayout(QHBoxLayout())
        box6a.layout().addWidget(self.file_path_input)
        box6a.layout().addWidget(browse_dir_button)
        box6.layout().addWidget(box6a)
        box6b = QGroupBox('File name'); box6b.setLayout(QHBoxLayout())
        box6b.layout().addWidget(self.file_name_input)
        box6b.layout().addWidget(browse_zarr_button)
        box6.layout().addWidget(box6b)        
        box6.layout().addWidget(from_source_button)
        box6.layout().addWidget(save3Dbtn)
        #box6.layout().addWidget(save23Dbtn, 3, 1, 1, 1)

        #self.layout().addWidget(box1)
        self.layout().addWidget(box6)
    
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
        try:
            raw = self.viewer.layers[self.active_image.currentText()].data
        except:
            print("Image layer not defined.")
            raw = None
        try:
            labels = self.viewer.layers[self.active_label.currentText()].data
        except:
            print("Labels layer not defined.")
            labels = None

        for (name, data, process_func) in (('raw', raw, self._process_raw), ('labeled', labels, self._process_labels)):
            if data is None:
                continue
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
            
            if not threeD and not twoD:
                if is_dask:
                    zarrfi[f'{name}'] = process_func(data.compute())
                else:
                    zarrfi[f'{name}'] = process_func(data)
                zarrfi[f'{name}'].attrs['offset'] = [0,]*3
                zarrfi[f'{name}'].attrs['resolution'] = [1,]*3

    def _process_raw(self, rawdata):
        return normalize(rawdata.astype(np.uint16), 
                maxval=(2**16-1)).astype(np.uint16)

    def _process_labels(self, labeldata):
        return labeldata.astype(np.int64)

    def _convert_dask(self, layer_name):
        try:
            self.viewer.layers[layer_name].data.chunks
        except:
            # not a dask
            return
        print('converting layer '+layer_name+' to numpy array')
        self.viewer.layers[layer_name].data = self.viewer.layers[layer_name].data.compute()
        self.labcheck.setCheckState(2)

    def _set_editable(self):
        if self.labcheck.isChecked():
            try:
                label = self.active_label.currentText()
            except:
                return
            self._convert_dask(self.active_label.currentText())

    def _new_pts(self):
        self.viewer.add_points(
                ndim=3, 
                face_color='magenta', 
                border_color='white',
                size=5,
                out_of_slice_display=True,
                opacity=0.7,
                symbol='x')
        # set active points layer to the new layer
        self.active_points.setCurrentText(self.viewer.layers[-1].name)
        
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
            print("Points layer not defined.")
            return
        
        ll = self.viewer.layers
        layer = img
        images = stack_to_images(layer, axis=1)
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

        if imgpath is not None and imgpath.endswith('.tif'):
            [z, y, x] = self._read_tiff_voxel_size(imgpath)
            self.xyres = x
            self.zres = z
        # TODO: add functionality for .czi or other formats?
        
    def _read_tiff_voxel_size(self, file_path):
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

    def _rxy(self):
        self.viewer.camera.angles = [self.viewer.camera.angles[0], 0, 90]
    
    def _auto_z(self):
        should_break=False
        try:
            pts = self.viewer.layers[self.active_points.currentText()]
        except:
            print("Points layer not defined.")
            should_break=True
        try:
            img = self.viewer.layers[self.active_image.currentText()]
        except:
            print("Image layer not defined.")
            should_break=True

        if not should_break:
            pts = self.viewer.layers[self.active_points.currentText()]
            img = self.viewer.layers[self.active_image.currentText()]
            # if len(pts.selected_data)>0:
            #     pt_list = pts.selected_data
            # else:
            pt_list = [i for i in range(len(pts.data))]
            for pt_id in pt_list: #s.data:
                coords = tuple(pts.data[pt_id].astype(int))
                maxz = np.argmax(img.data[:, coords[1], coords[2]])
                pts.data[pt_id, 0] = maxz
            pts.refresh()

    def _change_z(self, z):
        should_break=False
        try:
            self.viewer.layers[self.active_points.currentText()]
        except:
            print("Points layer not defined.")
            should_break=True
        if not should_break:
            for pt in self.viewer.layers[self.active_points.currentText()].selected_data:
                self.viewer.layers[self.active_points.currentText()].data[pt, 0] = self.viewer.layers[self.active_points.currentText()].data[pt, 0]+z
        
        self.viewer.layers[self.active_points.currentText()].refresh()

    def _connect_label_events(self):
        try:
            layer_name = self.active_label.currentText()
            if layer_name:
                self.viewer.layers[layer_name].events.labels_update.connect(self._set_max_label)
                self.viewer.layers[layer_name].events.selected_label.connect(self.labelbox.setValue)

        except:
            pass

    def _set_max_label(self):
        should_break=False
        try:
            self.viewer.layers[self.active_label.currentText()]
        except: # AttributeError:
            should_break=True
        if not should_break:
            try: #check if dask
                maxL = self.viewer.layers[self.active_label.currentText()].data.max().compute()
            except: #if not dask
                maxL = self.viewer.layers[self.active_label.currentText()].data.max()
            #maxL = self.viewer.layers[self.active_label.currentText()].data.max()
            self.labelbox.setMaximum(maxL)
            self.maxlab.setText("max label: "+str(maxL))
        
    def _remove_labels_wo_points(self):
        should_break=False
        try:
            labels = self.viewer.layers[self.active_label.currentText()]
        except:
            print("Labels layer not defined.")
            should_break = True
        try:
            points = self.viewer.layers[self.active_points.currentText()]
        except:
            print("Points layer not defined.")
            should_break = True
        if should_break:
            return
        
        mask = np.zeros_like(labels.data)
        for pt in points.data:
            selected_label = labels.data[tuple([int(i) for i in pt])]
            mask[labels.data==selected_label] = selected_label

        self.viewer.add_labels(mask, name='selected '+self.active_label.currentText())


    def _remove_label(self):
        should_break=False
        try:
            labels = self.viewer.layers[self.active_label.currentText()]
        except:
            print("Labels layer not defined.")
            should_break = True
        if should_break:
            return
        
        self._convert_dask(self.active_label.currentText())
        
        mask = labels.data == self.rem_label
        labels.data[mask] = 0
        labels.refresh()
        self._set_max_label()

    def _merge_labels(self):
        should_break=False
        try:
            labels = self.viewer.layers[self.active_label.currentText()]
            self._convert_dask(self.active_label.currentText())
        except:
            print("Labels layer not defined.")
            should_break = True
        try:
            labels2 = self.viewer.layers[self.active_merge_label.currentText()]
            self._convert_dask(self.active_merge_label.currentText())
        except:
            print("Merge labels layer not defined.")
            should_break = True
        if labels == labels2:
            print("Cannot merge labels with itself.")
            should_break = True
        if should_break:
            return

        max0 = labels.data.max()
        mask = labels2.data>0
        min0 = labels2.data[mask].min()
        if min0 <= max0:
            const = max0+1-min0
            labels2.data[mask] = labels2.data[mask]+const
        labels.data[mask] = labels2.data[mask][:]
        labels.refresh()
        self._set_max_label()
        self.viewer.layers.remove(self.active_merge_label.currentText())
        self.active_merge_label.setCurrentIndex(-1)
        try:
            n = self.viewer.layers[self.active_points.currentText()].data.shape[0]
            self.viewer.layers[self.active_points.currentText()].selected_data = [i for i in range(n)]
            self.viewer.layers[self.active_points.currentText()].remove_selected()
            #self.viewer.layers.remove(self.active_points.currentText())
            #del self.active_points.currentText()
        except:
            print("Points layer doesn't exist")
        
    def _labels2points(self):
        try:
            labels = self.viewer.layers[self.active_label.currentText()].data
        except:
            print("Labels layer not defined. Labels to points function exited.")
            return
        # cannot operate on dask...
        self._convert_dask(self.active_label.currentText())
        labels = self.viewer.layers[self.active_label.currentText()].data

        mask = labels>0
        com = center_of_mass(mask, labels=labels, index=np.unique(labels[mask]))
        self.viewer.add_points(com,             
                ndim=3, 
                face_color='green', 
                border_color='white',
                size=5,
                out_of_slice_display=True,
                opacity=0.7,
                symbol='x',
                name='points from '+self.active_label.currentText())
            
    def _points2labels(self):
        # TODO: add in auto-update of combos if points2labels function breaks due to missing layer information

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
        try:
            labels = self.viewer.layers[self.active_label.currentText()].data
            # check if dask:
            try:
                curr_n = labels.max().compute()
            except:
                curr_n = labels.max()
        except:
            curr_n = 0
        
        if should_break:
            print("Points to labels function exited.")
            return
        
        blur_sig = [self.blur_sig_z, self.blur_sig_xy, self.blur_sig_xy]
        if np.any(blur_sig):
            img_inv = gaussian_filter(np.min(img) + np.max(img) - img, blur_sig)
        else:
            img_inv = np.min(img) + np.max(img) - img

        markers = np.zeros(img.shape, dtype='int')
        mask = np.zeros(img.shape, dtype='bool')

        if self.snap_rad>0:
            self._snap_to_max

        # make markers:
        count = 1
        for pos in pts: 
            pos = np.round(pos).astype('int')
            markers[tuple(pos)] = count+curr_n
            count += 1

        # make mask:
        count = 1
        for pos in pts:
            pos = np.round(pos).astype('int')
            pointIntensity = img_inv[tuple(pos)]
            
            # find local min (inverted max) value:
            zrange, yrange, xrange, rel_pos = self._get_slices(self.max_rad_xy, self.max_rad_z, pos, img.shape)
            subim = img_inv[zrange, yrange, xrange]        
            local_min = np.min(subim) 
            # get local region to threshold, find local min value:
            zrange, yrange, xrange, rel_pos = self._get_slices(self.rad_xy, self.rad_z, pos, img.shape)
            subim = img_inv[zrange, yrange, xrange]
            local_max = np.max(subim) # background
            
            # threshold:
            thresh = self.threshold*local_min+(1-self.threshold)*local_max 
            #print(local_min, local_max, thresh)
            if thresh < pointIntensity:
                #print("threshold overriden for spot "+str(j)+" "+str(thresh)+" "+str(pointIntensity))
                thresh = self.threshold*pointIntensity+(1-self.threshold)*local_max 
            subim_mask = subim <= thresh
            
            # check for multiple objects:
            sublabels = label(subim_mask)
            if sublabels.max() > 1:
                wantLabel = sublabels[tuple(rel_pos)]
                subim_mask = sublabels == wantLabel
                
                # recheck max:
                thresh2 = self.threshold*np.min(subim[subim_mask])+(1-self.threshold)*local_max
                if thresh < thresh2:
                    subim_mask = subim <= thresh2
                    sublabels = label(subim_mask)
                    wantLabel = sublabels[tuple(rel_pos)]
                    subim_mask = sublabels == wantLabel
            
            if self.solidity_thresh > 0:
                pt_solidity = regionprops(subim_mask.astype('int'))[0].solidity
            
                if pt_solidity < self.solidity_thresh:
                    subim_mask = self._dist_watershed_sep(subim_mask, rel_pos)
            submask = mask[zrange, yrange, xrange]
            submask = np.logical_or(submask, subim_mask)

            mask[zrange, yrange, xrange] = submask
            
        if self.wshed_type=='Image':
            outlabels = watershed(img_inv, markers=markers, mask=mask)
        elif self.wshed_type=='Distance':
            outlabels = watershed(
                    distance_transform_edt(markers==0, sampling=[3,1,1]),
                    markers=markers,
                    mask=mask)
        else:
            print('invalid watershed type')
            return
        self.viewer.add_labels(outlabels, name='labels from '+self.active_points.currentText())
        
        self.active_merge_label.setCurrentText(self.viewer.layers[-1].name)

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
        
        blur_sig = [self.blur_sig_z, self.blur_sig_xy, self.blur_sig_xy]
        img_inv = gaussian_filter(np.min(img) + np.max(img) - img, blur_sig)
        count = 0

        for pos in pts:
            pos = np.round(pos).astype('int')
            zrange, yrange, xrange, rel_pos = self._get_slices(self.snap_rad, self.snap_rad, pos, img.shape)
            pointIntensity = img_inv[zrange, yrange, xrange]

            shift = np.unravel_index(np.argmin(pointIntensity), pointIntensity.shape)
            shift = np.asarray(shift)-self.snap_rad

            pos = (pos + shift).astype('int')
            pts[count] = pos
            count += 1
        self.viewer.layers[self.active_points.currentText()].refresh()


    def _get_slices(self, rad_xy, rad_z, loc, shape):
        x1 = max(loc[2] - rad_xy, 0) ; 
        x2 = min(loc[2] + rad_xy+1, shape[2]) ; 
        y1 = max(loc[1] - rad_xy, 0) ; 
        y2 = min(loc[1] + rad_xy+1, shape[1]) ; 
        z1 = max(loc[0] - rad_z, 0) ; 
        z2 = min(loc[0] + rad_z+1, shape[0]) ;
        relx = loc[2] - x1 ;
        rely = loc[1] - y1 ;
        relz = loc[0] - z1 ;
        
        return slice(z1,z2), slice(y1,y2), slice(x1,x2), [relz, rely, relx]
    
    def _dist_watershed_sep(self, mask, loc):
        dists = distance_transform_edt(mask, sampling=[3,1,1])
        pk_idx = peak_local_max(dists, labels=mask)
        pks = np.zeros_like(dists, dtype=bool)
        for pk in pk_idx:
            pks[tuple(pk)] = True
        pk_labels = label(pks)
        if pk_labels.max()>1:
            merged_peaks = center_of_mass(pks, pk_labels, index=range(1, pk_labels.max()+1))
            merged_peaks = np.round(merged_peaks).astype('int')
            
            markers = np.zeros_like(mask, dtype='int')
            for i in range(merged_peaks.shape[0]):
                markers[merged_peaks[i,0], merged_peaks[i,1], merged_peaks[i,2]] = i+1

            labels = watershed(-dists, markers=markers, mask=mask)
            wantLabel = labels[loc[0], loc[1], loc[2]] 
            mask_out = labels == wantLabel
        
        else:
            mask_out = mask
        
        return mask_out
    
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
        #print('setting attribute', value, attrname)
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

def normalize(data, maxval=1., dtype=np.uint16):
    data = data.astype(dtype)
    data_norm = data - data.min()
    scale_fact = maxval/data_norm.max()
    data_norm = data_norm * scale_fact
    return data_norm.astype(dtype)
