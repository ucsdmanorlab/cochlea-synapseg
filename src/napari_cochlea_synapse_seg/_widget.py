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

from ._GTwidget import GTWidget
from ._predwidget import PredWidget
from ._reader import napari_get_reader
import os
import numpy as np
import zarr
import tifffile

if TYPE_CHECKING:
    import napari
                
class SynapSegWidget(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.init_ui()
    
    def init_ui(self):
        self.setLayout(QVBoxLayout())
        tab_widget = QTabWidget()
        tab1 = GTWidget(viewer=self.viewer)
        tab2 = PredWidget(viewer=self.viewer)
        #QWidget()

        tab_widget.addTab(tab1, "Ground Truth")
        tab_widget.addTab(tab2, "Predict")
        self.layout().addWidget(tab_widget)

