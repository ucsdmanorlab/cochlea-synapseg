"""
This module provides a custom QWidget class (GTWidget) for use with the napari viewer.
It includes various functionalities to display synapse images, edit and create point and
label annotations, interconvert between points and labels, and save data in as .zarr.
"""
from typing import TYPE_CHECKING

from qtpy.QtWidgets import QTabWidget, QScrollArea, QScrollBar, QVBoxLayout, QWidget
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
        tab_widget = QTabWidget()
        tab1 = self._init_scroll(GTWidget(viewer=self.viewer))
        tab2 = self._init_scroll(PredWidget(viewer=self.viewer))
        
        tab_widget.addTab(tab1, "Ground Truth")
        tab_widget.addTab(tab2, "Predict")
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(tab_widget)

    def _init_scroll(self, widget):
        #scrollbox = QVBoxLayout(self)
        #self.setLayout(scrollbox)

        scrollarea = QScrollArea(self)
        scrollarea.setWidgetResizable(True)

        #self.layout().addWidget(scrollarea)
        
        scrollContent = QWidget(scrollarea)
        scrollLayout = QVBoxLayout(scrollContent)
        scrollContent.setLayout(scrollLayout)

        scrollLayout.addWidget(widget)
        scrollarea.setWidget(scrollContent)

        return scrollarea


