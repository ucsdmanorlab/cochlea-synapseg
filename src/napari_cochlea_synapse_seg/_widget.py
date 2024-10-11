"""
This module contains two napari widgets:

- 1. Point loader widget, with functionality related to 
    loading points (convert to real pixel units,   Python function flagged with `autogenerate: true`
    in the plugin manifest. Type annotations are used by
    magicgui to generate widgets for each parameter. Best
    suited for simple processing tasks - usually taking
    in and/or returning a layer.
- a `magic_factory` decorated function. The `magic_factory`
    decorator allows us to customize aspects of the resulting
    GUI, including the widgets associated with each parameter.
    Best used when you have a very simple processing task,
    but want some control over the autogenerated widgets. If you
    find yourself needing to define lots of nested functions to achieve
    your functionality, maybe look at the `Container` widget!
- a `magicgui.widgets.Container` subclass. This provides lots
    of flexibility and customization options while still supporting
    `magicgui` widgets and convenience methods for creating widgets
    from type annotations. If you want to customize your widgets and
    connect callbacks, this is the best widget option for you.
- a `QWidget` subclass. This provides maximal flexibility but requires
    full specification of widget layouts, callbacks, events, etc.

References:
- Widget specification: https://napari.org/stable/plugins/guides.html?#widgets
- magicgui docs: https://pyapp-kit.github.io/magicgui/

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from qtpy.QtWidgets import QLabel, QCheckBox, QSpinBox, QDoubleSpinBox, QGroupBox, QHBoxLayout, QVBoxLayout, QGridLayout, QPushButton, QWidget, QFileDialog, QComboBox
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

from napari_builtins.io._read import magic_imread

        
class ZarrWidget(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.file_path = ''
        # Data loader box
        zarrbtn = QPushButton("Load zarr")
        
        savebox = QGroupBox('Save zarr')

        save3Dbtn = QPushButton("Save 3D only")
        save23Dbtn = QPushButton("Save 2D and 3D")

        zarrbtn.clicked.connect(self._choose_zarr)
        save3Dbtn.clicked.connect(lambda: self._save_zarr(threeD=True, twoD=False))
        save23Dbtn.clicked.connect(lambda: self._save_zarr(threeD=True, twoD=True))
        savebox.setLayout(QHBoxLayout())
        savebox.layout().addWidget(save3Dbtn)
        savebox.layout().addWidget(save23Dbtn)

        self.layout().addWidget(zarrbtn)
        
        self.layout().addWidget(savebox)

    def _choose_zarr(self):
        self.file_path = QFileDialog.getExistingDirectory(
                self,
                caption="Choose .zarr with 3d/raw and 3d/labeled inside",
                )
        image = magic_imread(os.path.join(self.file_path,'3d/raw'))
        labels = magic_imread(os.path.join(self.file_path,'3d/labeled'))

        self.viewer.dims.ndisplay = 3
        n = len(self.viewer.layers)
        self.viewer.add_layer(image)
        self.viewer.layers[n].reset_contrast_limits()
        self.viewer.add_layer(labels)
        self.viewer.layers[n+1].brush_size=4

    def _save_zarr(self, threeD=True, twoD=False):
        self.file_path = QFileDialog.getExistingDirectory(
                self,
                caption="Choose .zarr with 3d/raw and 3d/labeled inside",
                )

    def _save_in_place(self):
        zarrfi = zarr.open(self.file_path)
    
        ## make edits
        labels = self.viewer.layers[self.active_label].data
        zarrfi[os.path.join('3d','labeled')] = labels
        zarrfi[os.path.join('3d','labeled')].attrs['offset'] = [0,]*3
        zarrfi[os.path.join('3d','labeled')].attrs['resolution'] = [1,]*3
        
        for z in range(labels.shape[0]):
            zarrfi[os.path.join('2d','labeled', str(z))] = np.expand_dims(labels[z], axis=0)
            zarrfi[os.path.join('2d','labeled', str(z))].attrs['offset'] = [0,]*2
            zarrfi[os.path.join('2d','labeled', str(z))].attrs['resolution'] = [1,]*2
        
                
class GTWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.setLayout(QVBoxLayout())
        self.file_path = ''
        self.labels_editable = False
        self.xyres = 1
        self.zres = 1

        # Active layers box ##############################################################
        #box1 = QGroupBox('Active layers')
        imgcombo = QComboBox() 
        labcombo = QComboBox()
        ptscombo = QComboBox()

        img_refreshbtn = QPushButton("\u27F3"); img_refreshbtn.setToolTip("Refresh")
        lab_refreshbtn = QPushButton("\u27F3"); lab_refreshbtn.setToolTip("Refresh")
        pts_refreshbtn = QPushButton("\u27F3"); pts_refreshbtn.setToolTip("Refresh")

        self.labcheck = QCheckBox("Make labels editable")
        self.labcheck.setTristate(False); self.labcheck.setCheckState(self.labels_editable)

        imgcombo.currentTextChanged.connect(lambda name: _update_attr(self, name, 'active_image'))
        labcombo.currentTextChanged.connect(self._set_active_label)
        ptscombo.currentTextChanged.connect(lambda name: _update_attr(self, name, 'active_points'))
        
        self.labcheck.stateChanged.connect(self._set_editable)
        img_refreshbtn.clicked.connect(lambda: _update_combos(self, imgcombo, 'Image'))
        lab_refreshbtn.clicked.connect(lambda: _update_combos(self, labcombo, 'Labels'))
        pts_refreshbtn.clicked.connect(lambda: _update_combos(self, ptscombo, 'Points'))
        _update_combos(self, imgcombo, 'Image', set_index=-1) 
        _update_combos(self, ptscombo, 'Points', set_index=-1)
        _update_combos(self, labcombo, 'Labels', set_index=-1)
        
        # set layout:
        # layers_gbox = QGridLayout()
        # layers_gbox.addWidget(QLabel('image layer:'), 1, 0) ; layers_gbox.addWidget(imgcombo, 1, 1)
        # layers_gbox.addWidget(img_refreshbtn, 1, 2)
        # layers_gbox.addWidget(QLabel('points layer:'), 2, 0) ; layers_gbox.addWidget(ptscombo, 2, 1)
        # layers_gbox.addWidget(pts_refreshbtn, 2, 2)      
        #   
        # layers_gbox.addWidget(QLabel('labels layer:'), 3, 0) ; layers_gbox.addWidget(labcombo, 3, 1)
        # layers_gbox.addWidget(lab_refreshbtn, 3, 2)
        # layers_gbox.addWidget(self.labcheck, 4, 0)
        
        # box1.setLayout(layers_gbox)
        
        # Image tools box ################################################################
        box2 = QGroupBox('Image tools')

        xyresbox = QDoubleSpinBox()
        zresbox  = QDoubleSpinBox()
        splitbtn = QPushButton('Split channels')
        splitbtn.clicked.connect(self._split_channels); splitbtn.clicked.connect(lambda: _update_combos(self, imgcombo, 'Image'))
        
        _setup_spin(self, xyresbox,  minval=0, val=self.xyres, step=0.05, attrname='xyres', dec=4, dtype=float)
        _setup_spin(self, zresbox,  minval=0, val=self.zres, step=0.05, attrname='zres', dec=4, dtype=float)
        imgcombo.currentTextChanged.connect(lambda: self._read_res())
        imgcombo.currentTextChanged.connect(lambda: xyresbox.setValue(self.xyres))
        imgcombo.currentTextChanged.connect(lambda: zresbox.setValue(self.zres))
        
        image_gbox = QGridLayout()
        image_gbox.addWidget(QLabel('image layer:'), 0, 0) ; image_gbox.addWidget(imgcombo, 0, 1)
        image_gbox.addWidget(img_refreshbtn, 0, 2)
        image_gbox.addWidget(QLabel('xy res:'), 1, 0) 
        image_gbox.addWidget(xyresbox, 1, 1)
        image_gbox.addWidget(QLabel('z res:'), 2, 0) 
        image_gbox.addWidget(zresbox, 2, 1)
        image_gbox.addWidget(splitbtn, 3, 0, 1, 2)

        box2.setLayout(image_gbox)

        # Point tools box ################################################################
        box3 = QGroupBox('Points tools')
        scalepts = QPushButton('real -> pixel units')
        chbox = QSpinBox(); self.nch=1;  
        ch2zbtn = QPushButton('chan -> z convert')
        
        _setup_spin(self, chbox, minval=1, maxval=10, val=self.nch, attrname='nch')
        ch2zbtn.clicked.connect(self._convert_ch2z)
        scalepts.clicked.connect(self._scale_points)

        points_gbox = QGridLayout()
        points_gbox.addWidget(QLabel('points layer:'), 0, 0) ; points_gbox.addWidget(ptscombo, 0, 1)
        points_gbox.addWidget(pts_refreshbtn, 0, 2)
        points_gbox.addWidget(scalepts, 1, 0, 1, 2)
        points_gbox.addWidget(chbox, 2, 0)
        points_gbox.addWidget(ch2zbtn, 2, 1)
        box3.setLayout(points_gbox)

        # Label tools box ################################################################
        box4 = QGroupBox('Labels tools')
        self.labelbox = QSpinBox(); self.rem_label = 1
        removebtn = QPushButton("Remove label")
        self.maxlab = QLabel("max label: ")
        labn_refreshbtn = QPushButton("\u27F3"); labn_refreshbtn.setToolTip("Refresh")
        labn_refreshbtn.clicked.connect(self._set_max_label)
        labcombo.currentTextChanged.connect(self._set_max_label)

        lab2combo = QComboBox(); 
        lab2_refreshbtn = QPushButton("\u27F3"); lab2_refreshbtn.setToolTip("Refresh")
        mlsbtn = QPushButton("Merge labels")        
        l2pbtn = QPushButton("Labels to points")

        lab2combo.currentTextChanged.connect(lambda name: _update_attr(self, name, 'active_label2'))
        lab2_refreshbtn.clicked.connect(lambda: _update_combos(self,lab2combo, 'Labels'))
        mlsbtn.clicked.connect(self._merge_labels)
        mlsbtn.clicked.connect(lambda: _update_combos(self,lab2combo, 'Labels', set_index=-1))
        removebtn.clicked.connect(self._remove_label)
        l2pbtn.clicked.connect(self._labels2points)
        l2pbtn.clicked.connect(lambda: _update_combos(self, ptscombo, 'Points'))
        _setup_spin(self, self.labelbox, minval=1, maxval=1000, val=self.rem_label, attrname='rem_label')

        labels_gbox = QGridLayout()
        labels_gbox.addWidget(QLabel('labels layer:'), 0, 0) ; labels_gbox.addWidget(labcombo, 0, 1)
        labels_gbox.addWidget(lab_refreshbtn, 0, 2)
        labels_gbox.addWidget(self.labcheck, 1, 0)
        labels_gbox.addWidget(self.labelbox, 2, 0)
        labels_gbox.addWidget(removebtn, 2, 1)
        labels_gbox.addWidget(self.maxlab, 3, 0)
        labels_gbox.addWidget(labn_refreshbtn, 3, 2)
        labels_gbox.addWidget(QLabel('merge labels:'), 4, 0); 
        labels_gbox.addWidget(lab2combo, 4, 1); 
        labels_gbox.addWidget(lab2_refreshbtn, 5, 2)
        labels_gbox.addWidget(mlsbtn, 5, 0, 1, 2)
        labels_gbox.addWidget(l2pbtn, 6, 0, 1, 2)

        box4.setLayout(labels_gbox)

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

        box5b = QGroupBox('Advanced settings')
        radxybox = QSpinBox(); 
        radzbox = QSpinBox(); 
        snapbox = QSpinBox(); 
        mradxybox = QSpinBox() ; 
        mradzbox = QSpinBox(); 
        sigxybox = QDoubleSpinBox(); 
        sigzbox = QDoubleSpinBox(); 
        _setup_spin(self, radxybox,  minval=1, suff=' px', val=self.rad_xy, attrname='rad_xy')
        _setup_spin(self, radzbox,   minval=0, suff=' px', val=self.rad_z, attrname='rad_z')
        _setup_spin(self, snapbox,   minval=0, suff=' px', val=self.snap_rad, attrname='snap_rad')
        _setup_spin(self, mradxybox, minval=0, suff=' px', val=self.max_rad_xy, attrname='max_rad_xy')
        _setup_spin(self, mradzbox,  minval=0, suff=' px', val=self.max_rad_z, attrname='max_rad_z')
        _setup_spin(self, sigxybox,  minval=0, suff=' px', val=self.blur_sig_xy, step=0.1, attrname='blur_sig_xy', dtype=float)
        _setup_spin(self, sigzbox,   minval=0, suff=' px', val=self.blur_sig_z, step=0.1, attrname='blur_sig_z', dtype=float)
        

        ptsbtn.clicked.connect(self._new_pts)
        ptsbtn.clicked.connect(lambda: _update_combos(self, ptscombo,'Points', set_index=-1))
        p2mbtn.clicked.connect(self._auto_z)
        rxybtn.clicked.connect(self._rxy)
        zbox.valueChanged[int].connect(self._change_z)
        zbox.valueChanged[int].connect(lambda: zbox.setValue(0))
        snapbtn.clicked.connect(self._snap_to_max)
        p2lbtn.clicked.connect(self._points2labels)
        p2lbtn.clicked.connect(lambda: _update_combos(self,labcombo, 'Labels'))
        p2lbtn.clicked.connect(lambda: _update_combos(self,lab2combo, 'Labels', set_index=-1))
        
        box5b.setVisible(False)
        advbtn.toggled.connect(box5b.setVisible)
        
        _update_combos(self, imgcombo, 'Image')
        _update_combos(self, labcombo, 'Labels')
        _update_combos(self, ptscombo, 'Points')
        _update_combos(self, lab2combo,'Labels')

        gbox5b = QGridLayout()
        gbox5b.addWidget(QLabel('threshold xy rad:'), 0, 0); gbox5b.addWidget(radxybox, 0, 1)
        gbox5b.addWidget(QLabel('threshold z rad:'), 1, 0); gbox5b.addWidget(radzbox, 1, 1)
        gbox5b.addWidget(QLabel('snap to max rad:'), 2, 0); gbox5b.addWidget(snapbox, 2, 1)
        gbox5b.addWidget(QLabel('local max xy rad:'), 3, 0); gbox5b.addWidget(mradxybox, 3, 1)
        gbox5b.addWidget(QLabel('local max z rad:'), 4, 0); gbox5b.addWidget(mradzbox, 4, 1)
        gbox5b.addWidget(QLabel('gaussian xy rad:'), 5, 0); gbox5b.addWidget(sigxybox, 5, 1)
        gbox5b.addWidget(QLabel('gaussian z rad:'), 6, 0); gbox5b.addWidget(sigzbox, 6, 1)
        box5b.setLayout(gbox5b)

        p2l_gbox = QGridLayout()
        p2l_gbox.addWidget(ptsbtn, 0, 0, 1, 2)
        # gbox2.addWidget(QLabel('points layer:'), 1, 0) ; gbox2.addWidget(ptscombo, 1, 1)
        # gbox2.addWidget(pts_refreshbtn, 1, 2)
        p2l_gbox.addWidget(rxybtn, 1, 0)
        p2l_gbox.addWidget(p2mbtn, 1, 1)
        p2l_gbox.addWidget(QLabel('manually edit z:'), 2, 0)
        p2l_gbox.addWidget(zbox, 2, 1)
        p2l_gbox.addWidget(snapbtn, 3, 0, 1, 2)  
        p2l_gbox.addWidget(p2lbtn, 4, 0, 1, 2)  
        p2l_gbox.addWidget(advbtn, 5, 0, 1, 2)
        p2l_gbox.addWidget(box5b, 6, 0, 1, 2)
        
        box5.setLayout(p2l_gbox)

        box6 = QGroupBox('Save zarr')
        save3Dbtn = QPushButton("Save 3D only")
        save23Dbtn = QPushButton("Save 2D and 3D")

        save3Dbtn.clicked.connect(lambda: self._save_zarr(threeD=True, twoD=False))
        save23Dbtn.clicked.connect(lambda: self._save_zarr(threeD=True, twoD=True))
        box6.setLayout(QHBoxLayout())
        box6.layout().addWidget(save3Dbtn)
        box6.layout().addWidget(save23Dbtn)

        #self.layout().addWidget(box1)
        self.layout().addWidget(box2)
        self.layout().addWidget(box3)
        self.layout().addWidget(box5)
        self.layout().addWidget(box4)
        self.layout().addWidget(box6)

    def _save_zarr(self, threeD=True, twoD=False):
        fileName, _ = QFileDialog.getSaveFileName(self, "Save to .zarr",
                                       filter="Zarrs (*.zarr)")
        print(fileName)
        if len(fileName)>0:
            if not fileName.endswith(".zarr"):
                fileName = fileName + ".zarr"
        else:
            return

        # TODO: add dialog box to warn if overwriting a file

        zarrfi = zarr.open(fileName)

        raw = self.viewer.layers[self.active_image].data
        labels = self.viewer.layers[self.active_label].data

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

    def _set_active_label(self, name):     
        self.active_label = name
        if self.labels_editable:
            self._convert_dask(name)
    
    def _convert_dask(self, layer_name):
        try:
            self.viewer.layers[layer_name].data.chunks
        except:
            # not a dask
            return
        print('converting layer '+layer_name+' to numpy array')
        self.viewer.layers[layer_name].data = self.viewer.layers[layer_name].data.compute()
        self.labels_editable = True
        self.labcheck.setCheckState(2)

    def _set_editable(self, state):
        self.labels_editable = bool(state)
        if self.labels_editable:
            try:
                label = self.active_label
            except AttributeError:
                return
            self._convert_dask(self.active_label)

    def _new_pts(self):
        self.viewer.add_points(
                ndim=3, 
                face_color='magenta', 
                border_color='white',
                size=5,
                out_of_slice_display=True,
                opacity=0.7,
                symbol='x')
        
    def _convert_ch2z(self):
        try:
            pts = self.viewer.layers[self.active_points]
        except AttributeError:
            print("Points layer not defined.")
            return
        
        pts.data[:,0] = pts.data[:,0]/self.nch
        pts.refresh()
    
    def _split_channels(self):
        try:
            img = self.viewer.layers[self.active_image]
        except AttributeError:
            print("Points layer not defined.")
            return
        
        ll = self.viewer.layers
        layer = img
        images = stack_to_images(layer, axis=1)
        ll.remove(layer)
        ll.extend(images)
        
    def _scale_points(self):
        try:
            pts = self.viewer.layers[self.active_points]
        except AttributeError:
            print("Points layer not defined.")
            return
        
        pts.data[:,0] = pts.data[:,0]/self.zres
        pts.data[:,1] = pts.data[:,1]/self.xyres
        pts.data[:,2] = pts.data[:,2]/self.xyres
        
        pts.refresh()

    def _read_res(self):
        try:
            img = self.viewer.layers[self.active_image]
        except AttributeError:
            return
        imgpath = img.source.path

        if imgpath.endswith('.tif'):
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
            pts = self.viewer.layers[self.active_points]
        except AttributeError:
            print("Points layer not defined.")
            should_break=True
        try:
            img = self.viewer.layers[self.active_image]
        except AttributeError:
            print("Image layer not defined.")
            should_break=True

        if not should_break:
            pts = self.viewer.layers[self.active_points]
            img = self.viewer.layers[self.active_image]
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
            self.viewer.layers[self.active_points]
        except AttributeError:
            print("Points layer not defined.")
            should_break=True
        if not should_break:
            for pt in self.viewer.layers[self.active_points].selected_data:
                self.viewer.layers[self.active_points].data[pt, 0] = self.viewer.layers[self.active_points].data[pt, 0]+z
        
        self.viewer.layers[self.active_points].refresh()

    def _set_max_label(self):
        should_break=False
        try:
            self.viewer.layers[self.active_label]
        except: # AttributeError:
            should_break=True
        if not should_break:
            try: #check if dask
                maxL = self.viewer.layers[self.active_label].data.max().compute()
            except: #if not dask
                maxL = self.viewer.layers[self.active_label].data.max()
            #maxL = self.viewer.layers[self.active_label].data.max()
            self.labelbox.setMaximum(maxL)
            self.maxlab.setText("max label: "+str(maxL))
        
    def _remove_label(self):
        should_break=False
        try:
            labels = self.viewer.layers[self.active_label]
        except AttributeError:
            print("Labels layer not defined.")
            should_break = True
        if should_break:
            return
        
        self._convert_dask(self.active_label)
        
        mask = labels.data == self.rem_label
        labels.data[mask] = 0
        labels.refresh()

    def _merge_labels(self):
        should_break=False
        try:
            labels = self.viewer.layers[self.active_label]
            self._convert_dask(self.active_label)
        except AttributeError:
            print("Labels layer not defined.")
            should_break = True
        try:
            labels2 = self.viewer.layers[self.active_label2]
            self._convert_dask(self.active_label2)
        except AttributeError:
            print("Merge labels layer not defined.")
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
        self.viewer.layers.remove(self.active_label2)
        del self.active_label2
        try:
            n = self.viewer.layers[self.active_points].data.shape[0]
            self.viewer.layers[self.active_points].selected_data = [i for i in range(n)]
            self.viewer.layers[self.active_points].remove_selected()
            #self.viewer.layers.remove(self.active_points)
            #del self.active_points
        except:
            print("Points layer doesn't exist")
        
    def _labels2points(self):
        try:
            labels = self.viewer.layers[self.active_label].data
        except AttributeError:
            print("Labels layer not defined. Labels to points function exited.")
            return
        # cannot operate on dask...
        self._convert_dask(self.active_label)
        labels = self.viewer.layers[self.active_label].data

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
                name='points from '+self.active_label)
            
    def _points2labels(self):
        should_break=False
        try:
            pts = self.viewer.layers[self.active_points].data
        except AttributeError:
            print("Points layer not defined.")
            should_break=True
        try:
            img = self.viewer.layers[self.active_image].data
        except AttributeError:
            print("Image layer not defined.")
            should_break=True
        try:
            labels = self.viewer.layers[self.active_label].data
            # check if dask:
            try:
                curr_n = labels.max().compute()
            except:
                curr_n = labels.max()
        except AttributeError:
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
            #tifffile.imwrite(os.path.join(label_dir,"subim_"+str(j)+".tif"), np.min(subim, axis=0))
            
            # threshold:
            thresh = 0.5*local_min + 0.5*local_max 
            if thresh < pointIntensity:
                #print("threshold overriden for spot "+str(j)+" "+str(thresh)+" "+str(pointIntensity))
                thresh = 0.5*local_max + 0.5*pointIntensity
            subim_mask = subim <= thresh
            
            # check for multiple objects:
            sublabels = label(subim_mask)
            if sublabels.max() > 1:
                wantLabel = sublabels[tuple(rel_pos)]
                subim_mask = sublabels == wantLabel
                
                # recheck max:
                thresh2 = 0.5*np.min(subim[subim_mask]) + 0.5*np.max(subim)
                if thresh < thresh2:
                    subim_mask = subim <= thresh2
                    sublabels = label(subim_mask)
                    wantLabel = sublabels[tuple(rel_pos)]
                    subim_mask = sublabels == wantLabel
            
            pt_solidity = regionprops(subim_mask.astype('int'))[0].solidity
            
            if pt_solidity < 0.8:
                subim_mask = self._dist_watershed_sep(subim_mask, rel_pos)
            submask = mask[zrange, yrange, xrange]
            submask = np.logical_or(submask, subim_mask)

            mask[zrange, yrange, xrange] = submask
            
        outlabels = watershed(img_inv, markers=markers, mask=mask)
        self.viewer.add_labels(outlabels, name='labels from '+self.active_points)

    def _snap_to_max(self):
        should_break=False
        try:
            pts = self.viewer.layers[self.active_points].data
        except AttributeError:
            print("Points layer not defined.")
            should_break=True
        try:
            img = self.viewer.layers[self.active_image].data
        except AttributeError:
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
        self.viewer.layers[self.active_points].refresh()


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
        dists = distance_transform_edt(mask, sampling=[4,1,1])
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
        if set_index is not None and abs(set_index) < count:
            combobox.setCurrentIndex(combolist.index(combolist[set_index])) #seems redundant but accomodates negative indices
        elif rememberID>=0 and rememberID < count:
            combobox.setCurrentIndex(rememberID)
