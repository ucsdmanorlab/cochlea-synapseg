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
        self.crop_size_spin.setValue(32)
        crop_layout.addWidget(self.crop_size_spin)
        layout.addLayout(crop_layout)

        crop_z_layout = QHBoxLayout()
        self.crop_size_z_spin = QSpinBox()
        self.crop_size_z_spin.setMinimum(1)
        self.crop_size_z_spin.setMaximum(512)
        self.crop_size_z_spin.setValue(16)
        crop_z_layout.addWidget(QLabel("Crop size (Z):"))
        crop_z_layout.addWidget(self.crop_size_z_spin)
        layout.addLayout(crop_z_layout)

        self.crop_btn = QPushButton("Create Montage")
        layout.addWidget(self.crop_btn)

        self.setLayout(layout)

        self.update_layer_choices()
        self.viewer.layers.events.inserted.connect(self.update_layer_choices)
        self.viewer.layers.events.removed.connect(self.update_layer_choices)

        self.crop_btn.clicked.connect(self.create_montage)

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

    def create_montage(self):
        img1 = self.viewer.layers[self.img1_combo.currentText()].data
        img2 = self.viewer.layers[self.img2_combo.currentText()].data
        labels = self.viewer.layers[self.labels_combo.currentText()].data
        crop_size = self.crop_size_spin.value()
        crop_size_z = self.crop_size_z_spin.value()

        props = regionprops(labels)
        crops1 = []
        crops2 = []

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
                    [(0, 0)] * (crop.ndim - 3) + [(0, pad_z), (0, pad_y), (0, pad_x)],
                    mode='constant'
                )
            else:
                pad_y = shape[0] - crop.shape[-2]
                pad_x = shape[1] - crop.shape[-1]
                return np.pad(
                    crop,
                    [(0, 0)] * (crop.ndim - 2) + [(0, pad_y), (0, pad_x)],
                    mode='constant'
                )

        if is_3d:
            target_shape = (crop_size_z, crop_size, crop_size)
            crops1 = [pad_crop(c, target_shape) for c in crops1]
            crops2 = [pad_crop(c, target_shape) for c in crops2]
            # Concatenate along x axis (last axis)
            # create grid of crops:
            nrows = int(np.ceil(np.sqrt(n)))
            ncols = int(np.ceil(n / nrows))

            def make_grid(crops, nrows, ncols, crop_shape):
                grid = np.zeros(
                    (crop_shape[0], nrows * crop_shape[1], ncols * crop_shape[2]),
                    dtype=crops[0].dtype
                )
                for idx, crop in enumerate(crops):
                    row = idx // ncols
                    col = idx % ncols
                    grid[
                        :,
                        row * crop_shape[1]:(row + 1) * crop_shape[1],
                        col * crop_shape[2]:(col + 1) * crop_shape[2]
                    ] = crop
                return grid

            montage1 = make_grid(crops1, nrows, ncols, target_shape)
            montage2 = make_grid(crops2, nrows, ncols, target_shape)
            # Optionally, stack them for display, or add separately:
            self.viewer.add_image(montage1, name="Crop Montage 1", colormap='green')
            self.viewer.add_image(montage2, name="Crop Montage 2", colormap='magenta', blending='additive')
            return  # prevent further code from running
        else:
            target_shape = (crop_size, crop_size)
            crops1 = [pad_crop(c, target_shape) for c in crops1]
            crops2 = [pad_crop(c, target_shape) for c in crops2]
            
            montage = np.stack([np.concatenate([c1, c2], axis=-1) for c1, c2 in zip(crops1, crops2)])

        self.viewer.add_image(montage, name="Crop Montage", colormap='gray')