from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QSpinBox, QHBoxLayout
import napari
import numpy as np
from skimage.measure import regionprops

class CropWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        layout = QVBoxLayout()

        # Layer selectors
        self.img1_combo = QComboBox()
        self.img2_combo = QComboBox()
        self.labels_combo = QComboBox()
        layout.addWidget(QLabel("Image Layer 1:"))
        layout.addWidget(self.img1_combo)
        layout.addWidget(QLabel("Image Layer 2:"))
        layout.addWidget(self.img2_combo)
        layout.addWidget(QLabel("Labels Layer:"))
        layout.addWidget(self.labels_combo)

        # Crop size
        crop_layout = QHBoxLayout()
        crop_layout.addWidget(QLabel("Crop size:"))
        self.crop_size_spin = QSpinBox()
        self.crop_size_spin.setMinimum(5)
        self.crop_size_spin.setMaximum(512)
        self.crop_size_spin.setValue(16)
        crop_layout.addWidget(self.crop_size_spin)
        layout.addLayout(crop_layout)

        crop_z_layout = QHBoxLayout()
        self.crop_size_z_spin = QSpinBox()
        self.crop_size_z_spin.setMinimum(1)
        self.crop_size_z_spin.setMaximum(512)
        self.crop_size_z_spin.setValue(8)
        crop_z_layout.addWidget(QLabel("Crop size (Z):"))
        crop_z_layout.addWidget(self.crop_size_z_spin)
        layout.addLayout(crop_z_layout)

        sort_layout = QHBoxLayout()
        sort_layout.addWidget(QLabel("Sort by:"))
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Label ID", "Size", "X", "Y", "Z", "Layer 1 Intensity", "Layer 2 Intensity"])
        sort_layout.addWidget(self.sort_combo)
        layout.addLayout(sort_layout)

        self.crop_btn = QPushButton("Create Montage")
        layout.addWidget(self.crop_btn)

        # add zoom to label functionality:
        self.label_id = QSpinBox()
        self.label_id.setMinimum(1)
        self.label_id.setMaximum(1000000)
        self.label_id.setValue(1)
        layout.addWidget(self.label_id)
        self.zoom_to_label_btn = QPushButton("Zoom to Label")
        layout.addWidget(self.zoom_to_label_btn)


        self.setLayout(layout)

        self.update_layer_choices()
        self.viewer.layers.events.inserted.connect(self.update_layer_choices)
        self.viewer.layers.events.removed.connect(self.update_layer_choices)

        self.crop_btn.clicked.connect(self.create_montage)
        self.zoom_to_label_btn.clicked.connect(self.zoom_to_label)

    def update_layer_choices(self, event=None):
        img_layers = [l.name for l in self.viewer.layers if l.__class__.__name__ == "Image"]
        label_layers = [l.name for l in self.viewer.layers if l.__class__.__name__ == "Labels"]

        img1_choice = self.img1_combo.currentText()
        img2_choice = self.img2_combo.currentText()
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

    def create_montage(self):
        img1 = self.viewer.layers[self.img1_combo.currentText()].data
        img2 = self.viewer.layers[self.img2_combo.currentText()].data
        labels = self.viewer.layers[self.labels_combo.currentText()].data
        crop_size = self.crop_size_spin.value()
        crop_size_z = self.crop_size_z_spin.value()

        props = regionprops(labels)
        crops1 = []
        crops2 = []
        sortby = []
        ids = []

        is_3d = labels.ndim == 3

        for prop in props:
            if is_3d:
                z, y, x = np.round(prop.centroid).astype(int)
                z0 = max(z - crop_size_z // 2, 0)
                z1 = min(z + crop_size_z // 2, img1.shape[-3])
                y0 = max(y - crop_size // 2, 0)
                y1 = min(y + crop_size // 2, img1.shape[-2])
                x0 = max(x - crop_size // 2, 0)
                x1 = min(x + crop_size // 2, img1.shape[-1])
                crop1 = img1[..., z0:z1, y0:y1, x0:x1]
                crop2 = img2[..., z0:z1, y0:y1, x0:x1]
            else:
                y, x = np.round(prop.centroid).astype(int)
                y0 = max(y - crop_size // 2, 0)
                y1 = min(y + crop_size // 2, img1.shape[-2])
                x0 = max(x - crop_size // 2, 0)
                x1 = min(x + crop_size // 2, img1.shape[-1])
                crop1 = img1[..., y0:y1, x0:x1]
                crop2 = img2[..., y0:y1, x0:x1]
            crops1.append(crop1)
            crops2.append(crop2)
            # sort by the selected property
            if self.sort_combo.currentText() == "Label ID":
                sortby.append(prop.label)
            elif self.sort_combo.currentText() == "X":
                sortby.append(x)
            elif self.sort_combo.currentText() == "Y":
                sortby.append(y)
            elif self.sort_combo.currentText() == "Z": 
                if is_3d:
                    sortby.append(z)
                else:
                    sortby.append(0)
            elif self.sort_combo.currentText() == "Size":
                sortby.append(prop.area)
            elif self.sort_combo.currentText() == "Layer 1 Intensity":
                sortby.append(np.mean(crop1))
            elif self.sort_combo.currentText() == "Layer 2 Intensity":
                sortby.append(np.mean(crop2))
            ids.append(prop.label)
        
        n = len(crops1)
        if n == 0:
            return

        def pad_crop(crop, shape):
            if is_3d:
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
            else:
                pad_y = shape[0] - crop.shape[-2]
                pad_x = shape[1] - crop.shape[-1]
                return np.pad(
                    crop,
                    ((pad_y//2, pad_y-pad_y//2),
                        (pad_x//2, pad_x-pad_x//2)),
                    mode='constant'
                )

        if is_3d:
            target_shape = (crop_size_z, crop_size, crop_size)
            crops1 = [pad_crop(c, target_shape) for c in crops1]
            crops2 = [pad_crop(c, target_shape) for c in crops2]
            # Sort by area:
            sorted_indices = np.argsort(sortby)
            if self.sort_combo.currentText() in ["Size", "Layer 1 Intensity", "Layer 2 Intensity"]:
                sorted_indices = sorted_indices[::-1]
            crops1 = [crops1[i] for i in sorted_indices]
            crops2 = [crops2[i] for i in sorted_indices]

            nrows = int(np.ceil(np.sqrt(n)))
            ncols = int(np.ceil(n / nrows))

            def make_grid(crops, nrows, ncols, crop_shape):
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

            montage1, cents = make_grid(crops1, nrows, ncols, target_shape)
            montage2, _ = make_grid(crops2, nrows, ncols, target_shape)
            ids = [ids[i] for i in sorted_indices]

            self.viewer.add_image(montage1, name="Montage Layer 1", colormap='green')
            self.viewer.add_image(montage2, name="Montage Layer 2", colormap='magenta', blending='additive')
            self.viewer.camera.center = [i//2 for i in montage1.data.shape]
            self.viewer.camera.angles = [0, 0, 90]

            self.viewer.add_points(
                cents,
                features={'id': ids}, 
                text={
                    'string': '{id}', 
                    'size': 10, 
                    'translation': [0, target_shape[1]//2-1, 0], 
                    'color':[1,1,1]}, 
                face_color=[0,0,0,0], 
                border_color=[0,0,0,0],
                name='Montage labels'
            )
            return  # prevent further code from running
        else:
            target_shape = (crop_size, crop_size)
            crops1 = [pad_crop(c, target_shape) for c in crops1]
            crops2 = [pad_crop(c, target_shape) for c in crops2]
            
            montage = np.stack([np.concatenate([c1, c2], axis=-1) for c1, c2 in zip(crops1, crops2)])

        self.viewer.add_image(montage, name="Crop Montage", colormap='gray')