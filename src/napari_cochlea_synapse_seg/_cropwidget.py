import numpy as np
import os
import matplotlib.pyplot as plt

from qtpy.QtWidgets import QGridLayout, QFileDialog,QDoubleSpinBox, QCheckBox, QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QSpinBox, QHBoxLayout, QGroupBox
from tifffile import imwrite

from skimage.filters import threshold_yen, threshold_li, threshold_otsu
from skimage.measure import regionprops
from skimage.restoration import rolling_ball
from skimage.segmentation import watershed
from scipy.ndimage import gaussian_filter
from .utils.post_proc import fit_line_length_in_box
from ._widget_utils import _setup_spin


class CropWidget(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.post_syn_labels = None

        layout = QVBoxLayout()

        layers_box = QGroupBox('Layers')
        layers_layout = QGridLayout()


        # Layer selectors
        self.img1_combo = QComboBox()
        self.img2_combo = QComboBox()
        self.labels_combo = QComboBox()
        renumber_btn = QPushButton("Renumber labels")
        renumber_btn.clicked.connect(self._renumber_labels)
        post_settings_btn = QPushButton("Post-synaptic detection options")
        post_settings_btn.setCheckable(True)
        
        layers_layout.addWidget(QLabel("Presynaptic Labels:"), 0, 0)
        layers_layout.addWidget(self.labels_combo, 0, 1)
        layers_layout.addWidget(renumber_btn, 1, 0, 1, 2)
        layers_layout.addWidget(QLabel("Presynaptic Image:"), 2, 0)
        layers_layout.addWidget(self.img1_combo, 2, 1)
        layers_layout.addWidget(QLabel("Postsynaptic Image:"), 3, 0)
        layers_layout.addWidget(self.img2_combo, 3, 1)
        layers_layout.addWidget(post_settings_btn, 4, 0, 1, 2)
        
        #############                
        # Advanced settings

        postSynBox = QGroupBox('Post-synaptic detection options')
        self.bkradbox = QSpinBox(); _setup_spin(self, self.bkradbox, minval=1, maxval=20, val=5, suff=' px')
        self.sigxybox = QDoubleSpinBox(); _setup_spin(self, self.sigxybox, minval=0, maxval=10, val=2.0, suff=' px', step=0.1, dtype=float)
        self.sigzbox = QDoubleSpinBox(); _setup_spin(self, self.sigzbox, minval=0, maxval=10, val=1.0, suff=' px', step=0.2, dtype=float)
        self.threshbox = QComboBox(); self.threshbox.addItem('Yen'); self.threshbox.addItem('Otsu'); self.threshbox.addItem('Li'); self.threshbox.setCurrentText('Yen')
        show_thresh = QPushButton('Show post-synaptic detection')
        show_thresh.clicked.connect(self._show_post_syn_detection)

        postSynBox.setVisible(False)
        post_settings_btn.toggled.connect(postSynBox.setVisible)
        
        gbox_postsyn = QGridLayout()
        gbox_postsyn.addWidget(QLabel('background rad:'), 0, 0); gbox_postsyn.addWidget(self.bkradbox, 0, 1)
        gbox_postsyn.addWidget(QLabel('gaussian xy rad:'), 1, 0); gbox_postsyn.addWidget(self.sigxybox, 1, 1)
        gbox_postsyn.addWidget(QLabel('gaussian z rad:'), 2, 0); gbox_postsyn.addWidget(self.sigzbox, 2, 1)
        gbox_postsyn.addWidget(self.threshbox, 3, 0); gbox_postsyn.addWidget(show_thresh, 3, 1)
        postSynBox.setLayout(gbox_postsyn)
        layers_layout.addWidget(postSynBox, 5, 0, 1, 2)

        layers_box.setLayout(layers_layout)
        layout.addWidget(layers_box)
        ##############

        # Crop size
        self.presyn_check = QCheckBox("Save presynaptic properties")
        self.montage_check = QCheckBox("Create montage")
        analysis_checks_layout = QHBoxLayout()
        analysis_checks_layout.addWidget(self.presyn_check)
        analysis_checks_layout.addWidget(self.montage_check)
        layout.addLayout(analysis_checks_layout)
        montage_settings_box = QGroupBox('Montage settings')
        montage_settings_box.setVisible(False)
        self.montage_check.toggled.connect(montage_settings_box.setVisible)
        
        montage_settings_layout = QGridLayout()
        montage_settings_layout.addWidget(QLabel("Crop size:"), 0, 0)
        self.crop_size_spin = QSpinBox()
        self.crop_size_spin.setMinimum(5)
        self.crop_size_spin.setMaximum(512)
        self.crop_size_spin.setValue(16)
        montage_settings_layout.addWidget(self.crop_size_spin, 0, 1)
        
        self.crop_size_z_spin = QSpinBox()
        self.crop_size_z_spin.setMinimum(1)
        self.crop_size_z_spin.setMaximum(512)
        self.crop_size_z_spin.setValue(8)
        montage_settings_layout.addWidget(QLabel("Crop size (Z):"), 1, 0)
        montage_settings_layout.addWidget(self.crop_size_z_spin, 1, 1)

        montage_settings_layout.addWidget(QLabel("Sort by:"), 2, 0)
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Label ID", "Size", "X", "Y", "Z", "Presynaptic Intensity", "Postsynaptic Intensity"])
        montage_settings_layout.addWidget(self.sort_combo, 2, 1)

        self.save_check = QCheckBox("Save crops to disk")
        self.save_check.setChecked(False)
        self.save_montage_check = QCheckBox("Save montage to disk")
        self.save_montage_check.setChecked(True)
        montage_settings_layout.addWidget(self.save_check, 3, 0)
        montage_settings_layout.addWidget(self.save_montage_check, 3, 1)
        montage_settings_box.setLayout(montage_settings_layout)
        layout.addWidget(montage_settings_box)

        # add save checkbox:
        self.save_dir = os.path.expanduser("~")
        
        self.crop_btn = QPushButton("Analyze")
        layout.addWidget(self.crop_btn)

        # add zoom to label functionality:
        self.zoom_box = QGroupBox('Montage controls')
        self.zoom_box.setVisible(False)
        self.label_id = QSpinBox()
        self.label_id.setMinimum(1)
        self.label_id.setMaximum(1000000)
        self.label_id.setValue(1)
        self.zoom_to_label_btn = QPushButton("Zoom to Label")
        self.zoom_to_montage_btn = QPushButton("Zoom to Montage")

        self.zoom_spin = QDoubleSpinBox()
        _setup_spin(self, self.zoom_spin, minval=0, maxval=1000, val=2, attrname='montage_zoom', dtype=float)
        zoom_current = QPushButton("Current")
        zoom_current.clicked.connect(lambda: self.zoom_spin.setValue(self.viewer.camera.zoom))
        
        zoom_box_layout = QVBoxLayout()
        labels_layout = QHBoxLayout()
        labels_layout.addWidget(QLabel("Label ID:"))
        labels_layout.addWidget(self.label_id)
        zoom_box_layout.addLayout(labels_layout)
        zoom_box_layout.addWidget(self.zoom_to_label_btn)
        montage_zoom_layout = QHBoxLayout()
        montage_zoom_layout.addWidget(QLabel("Montage Zoom:"))
        montage_zoom_layout.addWidget(self.zoom_spin)
        montage_zoom_layout.addWidget(zoom_current)
        zoom_box_layout.addLayout(montage_zoom_layout)
        zoom_box_layout.addWidget(self.zoom_to_montage_btn)
        self.zoom_box.setLayout(zoom_box_layout)
        layout.addWidget(self.zoom_box)

        self.setLayout(layout)

        self.update_layer_choices()
        self.viewer.layers.events.inserted.connect(self.update_layer_choices)
        self.viewer.layers.events.removed.connect(self.update_layer_choices)

        self.crop_btn.clicked.connect(self._analyze)
        self.zoom_to_label_btn.clicked.connect(self.zoom_to_label)
        self.zoom_to_montage_btn.clicked.connect(self.zoom_to_montage)

    def update_layer_choices(self, event=None):
        img_layers = [l.name for l in self.viewer.layers if l.__class__.__name__ == "Image"]
        label_layers = [l.name for l in self.viewer.layers if l.__class__.__name__ == "Labels"]

        img1_choice = self.img1_combo.currentText()
        img2_choice = self.img2_combo.currentText()
        if event and str(event.type) == 'removed' and str(event.value) == img2_choice:
            setattr(self, 'post_syn_labels', None)
        labels_choice = self.labels_combo.currentText()

        self.img1_combo.clear()
        self.img2_combo.clear()
        self.labels_combo.clear()
        self.img1_combo.addItems(img_layers)
        self.img2_combo.addItems(img_layers)
        self.labels_combo.addItems(label_layers)
        if img1_choice in img_layers:
            self.img1_combo.setCurrentText(img1_choice)
        if img2_choice in img_layers:
            self.img2_combo.setCurrentText(img2_choice)
        if labels_choice in label_layers:   
            self.labels_combo.setCurrentText(labels_choice)
        for layer in self.viewer.layers:
            try:
                layer.events.name.disconnect(self.update_layer_choices)
            except (TypeError, RuntimeError, ValueError):
                pass
            layer.events.name.connect(self.update_layer_choices)

    def _analyze(self):
        save_dir = self.get_save_directory()
        if ((self.montage_check.isChecked() and (self.save_check.isChecked() or self.save_montage_check.isChecked())) or self.presyn_check.isChecked()) and not save_dir:
            return
        if self.montage_check.isChecked():
            self.create_montage()
            self.zoom_box.setVisible(True)
        if self.presyn_check.isChecked():
            self._pre_syn_props()

    def _renumber_labels(self):
        labels_layer = self.viewer.layers[self.labels_combo.currentText()]
        labels_data = labels_layer.data
        props = regionprops(labels_data)
        new_labels = np.zeros_like(labels_data)
        for new_label, prop in enumerate(props, start=1):
            new_labels[labels_data == prop.label] = new_label
        labels_layer.data = new_labels
        self.post_syn_labels = None

    def _pre_syn_props(self):
        import pandas as pd
        from skimage.measure import regionprops_table

        # ADD CHECK TO CALCULATE IMAGE MEANS
        # ADD POSTSYN FUNCTIONALITY

        try:
            pre_syn_labels = self.viewer.layers[self.labels_combo.currentText()].data
            ctbp2 = self.viewer.layers[self.img1_combo.currentText()]
            ctbp2_data = ctbp2.data
            glur2_data = self.viewer.layers[self.img2_combo.currentText()].data
            pixel_size = ctbp2.scale if ctbp2.scale is not None else [1.0, 1.0, 1.0]
            if hasattr(pre_syn_labels, 'compute'):
                pre_syn_labels = pre_syn_labels.compute()
            if hasattr(ctbp2_data, 'compute'):
                ctbp2_data = ctbp2_data.compute()
        except KeyError:
            return
        syn_vol = regionprops(pre_syn_labels, intensity_image=ctbp2_data, spacing=pixel_size)
        avg_syn_volume = np.mean([s.area for s in syn_vol])
        avg_syn_intensity = np.mean([s.mean_intensity for s in syn_vol])
        syn_stats = regionprops_table(pre_syn_labels, intensity_image=ctbp2_data, spacing=pixel_size,
                                properties=['label', 'area', 'intensity_min', 'intensity_max', 'intensity_mean',
                                            'centroid', 'centroid_weighted', 'equivalent_diameter_area', 
                                            'axis_major_length', 'axis_minor_length', 'solidity'])
        syn_stats_df = pd.DataFrame(syn_stats)
        
        # post_syn = []
        # post_syn_int = []
        # for label_id in syn_stats['label']:
        #     if label_id == 0:
        #         continue
        #     post_syn_strict = np.any(post_syn_labels[pre_syn_labels==label_id])
        #     post_syn.append(post_syn_strict)

        #     row = syn_stats_df.loc[syn_stats_df['label'] == label_id, ['centroid-0', 'centroid-1', 'centroid-2']].iloc[0]
        #     z, y, x = row['centroid-0'], row['centroid-1'], row['centroid-2']

        #     centroid_px = (int(z/pixel_size[0]), int(y/pixel_size[1]), int(x/pixel_size[2]))
        #     local_crop = glur2_data[
        #         max(0, centroid_px[0]-2):centroid_px[0]+3,
        #         max(0, centroid_px[1]-4):centroid_px[1]+5,
        #         max(0, centroid_px[2]-4):centroid_px[2]+5
        #     ]
        #     post_syn_int.append(np.mean(local_crop))

        outfi = os.path.join(self.save_dir, f'pre_syn_stats.csv')
        syn_stats_df.to_csv(outfi, index=False)
    
        syn_stats_px = regionprops(pre_syn_labels)
        centroids_px = [syn_stat_px.centroid for syn_stat_px in syn_stats_px]

        line_length_um = fit_line_length_in_box(
            centroids=centroids_px,
            volume_shape=pre_syn_labels.shape,
            voxel_size=pixel_size,
        )

        if line_length_um is None or line_length_um == 0:
            ribbons_per_um = None
        else:
            ribbons_per_um = len(centroids_px) / line_length_um
        return

    def _show_post_syn_detection(self):
        labels = self._calc_post_syn_detection()
        if labels is None:
            return
        self.viewer.add_labels(
            labels, 
            name='Post-synaptic detection', 
        )
    
    def _calc_post_syn_detection(self):
        if self.post_syn_labels is not None:
            if self.bkradbox.value() == self.post_syn_labels['bk_rad'] and \
               self.sigxybox.value() == self.post_syn_labels['sig_xy'] and \
               self.sigzbox.value() == self.post_syn_labels['sig_z'] and \
               self.threshbox.currentText() == self.post_syn_labels['thresh'] and \
               self.img2_combo.currentText() == self.post_syn_labels['img']:
               print("Using cached post-synaptic labels")
               return self.post_syn_labels['labels']
        try:
            post_syn_layer = self.viewer.layers[self.img2_combo.currentText()]
        except KeyError:
            return
        
        bk = np.zeros_like(post_syn_layer.data)

        for (i, zslice) in enumerate(post_syn_layer.data):
            bk[i] = rolling_ball(zslice, radius=self.bkradbox.value())

        bk_sub = post_syn_layer.data - bk

        mask = gaussian_filter(bk_sub, sigma=(self.sigzbox.value()/2, self.sigxybox.value()/2, self.sigxybox.value()/2))
        smoothed = gaussian_filter(bk_sub, sigma=(self.sigzbox.value(), self.sigxybox.value(), self.sigxybox.value()))
        if self.threshbox.currentText() == 'Yen':
            thresh = threshold_yen(mask)
        elif self.threshbox.currentText() == 'Otsu':
            thresh = threshold_otsu(mask)
        elif self.threshbox.currentText() == 'Li':
            thresh = threshold_li(mask)
        mask = mask > thresh

        labels = watershed(-smoothed, mask=mask)
        self.post_syn_labels = {'bk_rad': self.bkradbox.value(),
                                'sig_xy': self.sigxybox.value(),
                                'sig_z': self.sigzbox.value(),
                                'thresh': self.threshbox.currentText(),
                                'img': self.img2_combo.currentText(),
                                'labels': labels}
        return labels

    def zoom_to_label(self):
        label_id = self.label_id.value()
        labels_layer = self.viewer.layers[self.labels_combo.currentText()]
        labels_data = labels_layer.data

        if label_id < 1 or label_id > labels_data.max():
            return

        props = regionprops(labels_data)
        for prop in props:
            if prop.label == label_id:
                if labels_data.ndim == 3:
                    z, y, x = np.round(prop.centroid).astype(int)
                    self.viewer.camera.zoom = 10
                    self.viewer.camera.center = (z, y, x)
                else:
                    y, x = np.round(prop.centroid).astype(int)
                    self.viewer.camera.zoom = 10
                    self.viewer.camera.center = (y, x)
                break
        # turn off montage layers:
        for layer in self.viewer.layers:
            if layer.name.startswith("Montage"):
                layer.visible = False
            if layer.name == self.labels_combo.currentText():
                layer.visible = True
            if layer.name == self.img1_combo.currentText():
                layer.visible = True
            if layer.name == self.img2_combo.currentText():
                layer.visible = True
    def zoom_to_montage(self):
        layer_names = [layer.name for layer in self.viewer.layers]
        montage_on = False
        for name in layer_names[::-1]:
            if name.startswith("Montage") and not montage_on:
                self.viewer.layers[name].visible = True
            else:
                self.viewer.layers[name].visible = False
            if name.startswith("Montage - Presynaptic") and not montage_on:
                self.viewer.camera.zoom = self.montage_zoom
                montage_layer = self.viewer.layers[name]
                self.viewer.camera.center = [i // 2 for i in montage_layer.data.shape]
                self.viewer.camera.angles = [0, 0, 90]
                montage_on = True            
        
    def get_save_directory(self):
        if ((self.montage_check.isChecked() and (self.save_check.isChecked() or self.save_montage_check.isChecked())) or self.presyn_check.isChecked()):
            save_dir = QFileDialog.getExistingDirectory(
                self, 
                "Select Directory to Save Outputs",
                self.save_dir
            )
            if save_dir:
                self.save_dir = save_dir
            return save_dir if save_dir else None
        return None

    def make_grid(self, crops, nrows, ncols, crop_shape):
            grid = np.zeros(
                (crop_shape[0], nrows * crop_shape[1], ncols * crop_shape[2]),
                dtype=crops[0].dtype
            )
            cents = []
            for idx, crop in enumerate(crops):
                row = idx // ncols
                col = idx % ncols
                grid[
                    :,
                    row * crop_shape[1]:(row + 1) * crop_shape[1],
                    col * crop_shape[2]:(col + 1) * crop_shape[2]
                ] = crop
                cents.append([crop_shape[0]/2, row * crop_shape[1] + crop_shape[1] // 2, col * crop_shape[2] + crop_shape[2] // 2])
            return grid, cents
    
    def create_montage(self):
        if (self.save_check.isChecked() or self.save_montage_check.isChecked()) and not self.save_dir:
            return  # User cancelled directory selection
        
        img1 = self.viewer.layers[self.img1_combo.currentText()].data
        img2 = self.viewer.layers[self.img2_combo.currentText()].data
        img2_labels = self._calc_post_syn_detection()#img2 > threshold_otsu(img2)

        labels = self.viewer.layers[self.labels_combo.currentText()].data
        crop_size = self.crop_size_spin.value()
        crop_size_z = self.crop_size_z_spin.value()

        props = regionprops(labels)
        crops1 = []
        crops2 = []
        sortby = []
        ids = []
        post_syn = []

        for prop in props:
            z, y, x = np.round(prop.centroid).astype(int)
            z0 = max(z - crop_size_z // 2, 0)
            z1 = min(z + crop_size_z // 2, img1.shape[-3])
            y0 = max(y - crop_size // 2, 0)
            y1 = min(y + crop_size // 2, img1.shape[-2])
            x0 = max(x - crop_size // 2, 0)
            x1 = min(x + crop_size // 2, img1.shape[-1])
            crop1 = img1[..., z0:z1, y0:y1, x0:x1]
            crop2 = img2[..., z0:z1, y0:y1, x0:x1]
            
            crops1.append(crop1)
            crops2.append(crop2)

            post_syn_strict = np.any(img2_labels[labels==prop.label])
            
            post_syn.append(post_syn_strict)
            
            # sort by the selected property
            if self.sort_combo.currentText() == "Label ID":
                sortby.append(prop.label)
            elif self.sort_combo.currentText() == "X":
                sortby.append(x)
            elif self.sort_combo.currentText() == "Y":
                sortby.append(y)
            elif self.sort_combo.currentText() == "Z": 
                sortby.append(z)
            elif self.sort_combo.currentText() == "Size":
                sortby.append(prop.area)
            elif self.sort_combo.currentText() == "Presynaptic Intensity":
                sortby.append(np.mean(crop1))
            elif self.sort_combo.currentText() == "Postsynaptic Intensity":
                sortby.append(np.mean(crop2))
            ids.append(prop.label)
        
        n = len(crops1)
        if n == 0:
            return

        def pad_crop(crop, shape):
            pad_z = shape[0] - crop.shape[-3]
            pad_y = shape[1] - crop.shape[-2]
            pad_x = shape[2] - crop.shape[-1]
            return np.pad(
                crop,
                ((pad_z//2, pad_z-pad_z//2),
                    (pad_y//2, pad_y-pad_y//2),
                    (pad_x//2, pad_x-pad_x//2)),
                mode='constant'
            )
            
        
        target_shape = (crop_size_z, crop_size, crop_size)
        crops1 = [pad_crop(c, target_shape) for c in crops1]
        crops2 = [pad_crop(c, target_shape) for c in crops2]
        # Sort by area:
        sorted_indices = np.argsort(sortby)

        if self.sort_combo.currentText() in ["Size", "Layer 1 Intensity", "Layer 2 Intensity"]:
            sorted_indices = sorted_indices[::-1]
        crops1 = [crops1[i] for i in sorted_indices]
        crops2 = [crops2[i] for i in sorted_indices]
        ids = [ids[i] for i in sorted_indices]
        post_syn = [post_syn[i] for i in sorted_indices]
        
        if self.save_check.isChecked():
            crop_dir = os.path.join(self.save_dir, "crops")
            os.makedirs(crop_dir, exist_ok=True)
            for (id, post, c1, c2) in zip(ids, post_syn, crops1, crops2):
                suff = '_pair' if post else '_orphan'
                filename = os.path.join(crop_dir, f"crop_{id}{suff}.tiff")
                # Create metadata for ImageJ composite mode with green/magenta LUTs
                metadata = {
                    'axes': 'ZCYX',
                    'mode': 'composite',
                }
                imwrite(filename, np.stack([c1, c2], axis=1), imagej=True, metadata=metadata)

        nrows = int(np.ceil(np.sqrt(n)))
        ncols = int(np.ceil(n / nrows))

        montage1, cents = self.make_grid(crops1, nrows, ncols, target_shape)
        montage2, _ = self.make_grid(crops2, nrows, ncols, target_shape)
        
        for layer in self.viewer.layers:
            if layer.name.startswith("Montage"):
                self.viewer.layers.remove(layer)
            else:
                layer.visible = False
        self.viewer.add_image(montage1, 
                              name="Montage - Presynaptic", 
                              colormap='magenta', 
                              contrast_limits=self.viewer.layers[self.img1_combo.currentText()].contrast_limits)
        self.viewer.add_image(montage2,
                              name="Montage - Postsynaptic", 
                              colormap='green', 
                              blending='additive',
                              contrast_limits=self.viewer.layers[self.img2_combo.currentText()].contrast_limits)
        # set viewer to 3D mode:
        self.viewer.dims.ndisplay = 3
        self.viewer.camera.center = [i//2 for i in montage1.data.shape]
        self.viewer.camera.angles = [0, 0, 90]
        self.viewer.camera.zoom = self.montage_zoom
        color_cycle = ['white', 'red'] if post_syn[0] else ['red', 'white']
        self.viewer.add_points(
            cents,
            features={'id': ids, 
                      'post_syn': post_syn}, 
            text={
                'string': '{id}', 
                'size': 10, 
                'translation': [0, target_shape[1]//2-1, 0], 
                'color': {'feature': 'post_syn', 'colormap': color_cycle}, #[1,1,1]}, 
            },
            face_color=[0,0,0,0], 
            border_color=[0,0,0,0],
            name='Montage labels',
            blending='additive',
        )
        if self.save_montage_check.isChecked():
            suff = '_montage.pdf'
            filename = os.path.join(self.save_dir, f"montage_{self.sort_combo.currentText()}{suff}")

            # create rgb image:
            rgb_montage = np.zeros((montage1.shape[1], montage1.shape[2], 3), dtype=montage1.dtype)
            rgb_montage[..., 0] = np.max(montage1, axis=0)  # Red channel (presynaptic)
            rgb_montage[..., 2] = np.max(montage1, axis=0)  # Blue + red = magenta (presynaptic)
            rgb_montage[..., 1] = np.max(montage2, axis=0)  # Green channel (postsynaptic)

            plt.figure(figsize=(10,10))
            plt.imshow(rgb_montage)
            plt.axis('off')
            
            # add text labels:
            for (id, post, cent) in zip(ids, post_syn, cents):
                color = 'white' if post else 'red'
                plt.text(cent[2], cent[1]+target_shape[1]//2-1, str(id), color=color, fontsize=12, ha='center', va='center')
            plt.tight_layout()
            plt.savefig(filename)
    
    def to_settings(self):
        """
        Serialize current widget settings to a dictionary.
        
        Returns:
            Dictionary of all persistent settings.
        """
        return {
            'save_presyn_props': self.presyn_check.isChecked(),
            'create_montage': self.montage_check.isChecked(),
            'crop_size_xy': self.crop_size_spin.value(),
            'crop_size_z': self.crop_size_z_spin.value(),
            'sort_by': self.sort_combo.currentText(),
            'save_montage_to_disk': self.save_montage_check.isChecked(),
            'save_crops_to_disk': self.save_check.isChecked(),
            'montage_zoom': self.montage_zoom,
            'bk_rad': self.bkradbox.value(),
            'sig_xy': self.sigxybox.value(),
            'sig_z': self.sigzbox.value(),
            'post_thresh_type': self.threshbox.currentText(),
        }
    
    def apply_settings(self, settings):
        """
        Restore widget settings from a dictionary.
        
        Args:
            settings: Dictionary of settings to apply.
        """
        if 'save_presyn_props' in settings:
            self.presyn_check.setChecked(settings['save_presyn_props'])
        if 'create_montage' in settings:
            self.montage_check.setChecked(settings['create_montage'])
        if 'crop_size_xy' in settings:
            self.crop_size_spin.setValue(settings['crop_size_xy'])
        if 'crop_size_z' in settings:
            self.crop_size_z_spin.setValue(settings['crop_size_z'])
        if 'sort_by' in settings:
            idx = self.sort_combo.findText(settings['sort_by'])
            if idx >= 0:
                self.sort_combo.setCurrentIndex(idx)
        if 'save_montage_to_disk' in settings:
            self.save_montage_check.setChecked(settings['save_montage_to_disk'])
        if 'save_crops_to_disk' in settings:
            self.save_check.setChecked(settings['save_crops_to_disk'])
        if 'montage_zoom' in settings:
            self.montage_zoom = settings['montage_zoom']
            self.zoom_spin.setValue(self.montage_zoom)
        if 'bk_rad' in settings:
            self.bkradbox.setValue(settings['bk_rad'])
        if 'sig_xy' in settings:
            self.sigxybox.setValue(settings['sig_xy'])
        if 'sig_z' in settings:
            self.sigzbox.setValue(settings['sig_z'])
        if 'post_thresh_type' in settings:
            idx = self.threshbox.findText(settings['post_thresh_type'])
            if idx >= 0:
                self.threshbox.setCurrentIndex(idx)