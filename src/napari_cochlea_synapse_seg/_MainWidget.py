"""
This module provides a custom QWidget class (GTWidget) for use with the napari viewer.
It includes various functionalities to display synapse images, edit and create point and
label annotations, interconvert between points and labels, and save data in as .zarr.
"""
from typing import TYPE_CHECKING
import atexit
from pathlib import Path

from qtpy.QtWidgets import QFileDialog, QTabWidget, QScrollArea, QSizePolicy, QVBoxLayout, QWidget, QPushButton, QHBoxLayout
from qtpy.QtCore import Qt, QTimer
from napari.utils.notifications import show_info, show_error

from ._GTWidget import GTWidget
from ._PredWidget import PredWidget
from ._PreprocessWidget import PreProcessWidget
from ._AnalyzeWidget import AnalyzeWidget
from ._settings import load_settings, save_settings


if TYPE_CHECKING:
    import napari
                
class SynapSegWidget(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        
        # Setup auto-save timer (debounced - waits 2 seconds after last change)
        self._autosave_timer = QTimer()
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.timeout.connect(self._save_settings_silent)
        self._autosave_timer.setInterval(2000)  # 2 seconds
        
        # Register cleanup on exit to save settings when napari closes
        atexit.register(self._save_settings_silent)
        
        self.init_ui()
    
    def init_ui(self):
        tab_widget = QTabWidget()
        
        # Store references to actual widget instances
        self.preprocess_widget = PreProcessWidget(viewer=self.viewer)
        self.gt_widget = GTWidget(viewer=self.viewer)
        self.pred_widget = PredWidget(viewer=self.viewer)
        self.crop_widget = AnalyzeWidget(viewer=self.viewer)
        
        tab0 = self._init_scroll(self.preprocess_widget)
        tab1 = self._init_scroll(self.gt_widget)
        tab2 = self._init_scroll(self.pred_widget)
        tab3 = self._init_scroll(self.crop_widget)
        
        tab_widget.addTab(tab0, "Preprocess")
        tab_widget.addTab(tab1, "Ground Truth")
        tab_widget.addTab(tab2, "Predict")
        tab_widget.addTab(tab3, "Analyze")
        
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(tab_widget)
        
        # Add save/load settings buttons
        settings_layout = QHBoxLayout()
        save_btn = QPushButton("Save Settings")
        load_btn = QPushButton("Load Settings")
        rest_btn = QPushButton("Restore Default Settings")
        save_btn.clicked.connect(self._save_settings)
        load_btn.clicked.connect(self._manual_load_settings)
        rest_btn.clicked.connect(self._restore_defaults)
        settings_layout.addWidget(save_btn)
        settings_layout.addWidget(load_btn)
        settings_layout.addWidget(rest_btn) 
        self.layout().addLayout(settings_layout)
        
        # Auto-load settings on startup
        self._load_settings()
        
        # Connect all widgets to trigger auto-save on changes
        self._connect_autosave()

    def _connect_autosave(self):
        """Connect all settings controls to trigger auto-save after changes."""
        # GT Widget controls
        self.gt_widget.z_scale.stateChanged.connect(self._trigger_autosave)
        self.gt_widget.threshbox.valueChanged.connect(self._trigger_autosave)
        self.gt_widget.radxybox.valueChanged.connect(self._trigger_autosave)
        self.gt_widget.radzbox.valueChanged.connect(self._trigger_autosave)
        self.gt_widget.mradxybox.valueChanged.connect(self._trigger_autosave)
        self.gt_widget.mradzbox.valueChanged.connect(self._trigger_autosave)
        self.gt_widget.sigxybox.valueChanged.connect(self._trigger_autosave)
        self.gt_widget.sigzbox.valueChanged.connect(self._trigger_autosave)
        self.gt_widget.solidbox.valueChanged.connect(self._trigger_autosave)
        self.gt_widget.thresholdbox.valueChanged.connect(self._trigger_autosave)
        self.gt_widget.wshedcombo.currentTextChanged.connect(self._trigger_autosave)
        self.gt_widget.file_path_input.textChanged.connect(self._trigger_autosave)
        
        # Pred Widget controls
        self.pred_widget.model_path_input.textChanged.connect(self._trigger_autosave)
        self.pred_widget.zarr_path_input.textChanged.connect(self._trigger_autosave)
        self.pred_widget.mask_thresh_box.valueChanged.connect(self._trigger_autosave)
        self.pred_widget.peak_thresh_box.valueChanged.connect(self._trigger_autosave)
        self.pred_widget.min_dist_box.valueChanged.connect(self._trigger_autosave)
        self.pred_widget.sig_xy_box.valueChanged.connect(self._trigger_autosave)
        self.pred_widget.sig_z_box.valueChanged.connect(self._trigger_autosave)
        self.pred_widget.size_filt_box.valueChanged.connect(self._trigger_autosave)
        
        # Crop Widget controls
        self.crop_widget.crop_size_spin.valueChanged.connect(self._trigger_autosave)
        self.crop_widget.crop_size_z_spin.valueChanged.connect(self._trigger_autosave)
        self.crop_widget.sort_combo.currentTextChanged.connect(self._trigger_autosave)
        self.crop_widget.save_montage_check.stateChanged.connect(self._trigger_autosave)
        self.crop_widget.save_check.stateChanged.connect(self._trigger_autosave)
    
    def _trigger_autosave(self):
        """Restart the autosave timer (debounced to avoid saving on every keystroke)."""
        self._autosave_timer.start()

    def _init_scroll(self, widget):
        scrollarea = QScrollArea(self)
        scrollarea.setWidgetResizable(True)

        scrollContent = QWidget(scrollarea)
        scrollLayout = QVBoxLayout(scrollContent)
        scrollContent.setLayout(scrollLayout)
        widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        scrollLayout.setAlignment(Qt.AlignTop)

        scrollLayout.addWidget(widget)
        scrollarea.setWidget(scrollContent)

        return scrollarea
    
    def _save_settings(self):
        """
        Save all widget settings to user specified file.
        """
        save_path = QFileDialog.getSaveFileName(
            self, 
            "Save Settings As", 
            str(Path.home() / "synapse_segmentation_settings.json"), 
            "JSON Files (*.json)"
        )[0]
        settings = {
            'version': '1.0',
            'GTWidget': self.gt_widget.to_settings(),
            'PredWidget': self.pred_widget.to_settings(),
            'CropWidget': self.crop_widget.to_settings(),
            'PreProcessWidget': {}  # Add when PreProcessWidget implements to_settings
        }
        
        if save_settings(settings, save_path):
            show_info("Settings saved successfully")
        else:
            show_error("Failed to save settings")
    
    def _save_settings_silent(self):
        """
        Save settings to default location, without showing notifications (for auto-save on exit).
        """
        try:
            settings = {
                'version': '1.0',
                'GTWidget': self.gt_widget.to_settings(),
                'PredWidget': self.pred_widget.to_settings(),
                'CropWidget': self.crop_widget.to_settings(),
                'PreProcessWidget': {}
            }
            save_settings(settings)
        except RuntimeError:
            # Qt widgets may have been deleted during shutdown, ignore
            pass

    def _manual_load_settings(self):
        load_path = QFileDialog.getOpenFileName(
            self, 
            "Load Settings", 
            str(Path.home() / "synapse_segmentation_settings.json"), 
            "JSON Files (*.json)"
        )[0]
        if load_path:
            self._load_settings(Path(load_path))

    def _load_settings(self, load_path=None):
        """
        Load settings, using user default path (on startup), 
        or user-specified path (manual load).
        """
        settings = load_settings(load_path)
        
        if 'GTWidget' in settings and settings['GTWidget']:
            self.gt_widget.apply_settings(settings['GTWidget'])
        
        if 'PredWidget' in settings and settings['PredWidget']:
            self.pred_widget.apply_settings(settings['PredWidget'])
        
        if 'CropWidget' in settings and settings['CropWidget']:
            self.crop_widget.apply_settings(settings['CropWidget'])
        
        # Only show info if settings were actually loaded (file existed)
        from ._settings import get_settings_path
        if get_settings_path().exists():
            show_info("Settings loaded")

    def _restore_defaults(self):
        # get path of current widget file
        default_path = Path(__file__).parent / 'default_settings.json'
        settings = load_settings(default_path)
        if default_path.exists():
            self.gt_widget.apply_settings(settings['GTWidget'])
            self.pred_widget.apply_settings(settings['PredWidget'])
            self.crop_widget.apply_settings(settings['CropWidget'])
            show_info("Default settings restored")
        else:
            show_error("Default settings file not found")
    
    def closeEvent(self, event):
        """
        Auto-save settings when widget is closed.
        """
        self._save_settings_silent()
        super().closeEvent(event)
