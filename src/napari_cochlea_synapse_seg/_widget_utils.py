from qtpy.QtGui import QFontMetrics
from qtpy.QtWidgets import QLabel, QDoubleSpinBox, QGroupBox, QVBoxLayout, QHBoxLayout, QCheckBox

def _wrap_builtin_method(method, *args, **kwargs):
    """
    Wrap a built-in method (e.g., QSpinBox.setValue) to make it compatible with 
    napari's event system, which tries to inspect method signatures.
    
    Args:
        method: Built-in method to wrap (e.g., spinbox.setValue)
        
    Returns:
        A wrapper function that napari can inspect and call with event parameter.
    """
    def wrapper(value, event=None):
        """Wrapper that accepts napari event parameter."""
        method(value)
    return wrapper

def _setup_spin(curr_class, spinbox, minval=None, maxval=None, suff=None, val=None, step=None, dec=None, attrname=None, dtype=int, connect_func=None):
    spinbox.setKeyboardTracking(False)
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
        spinbox.valueChanged[dtype].connect(lambda value: setattr(curr_class, attrname, value))
        setattr(curr_class, attrname, spinbox.value())
    if connect_func is not None:
        spinbox.valueChanged[dtype].connect(connect_func)

def _limitStretch(widget, max_chars=16):
    font_metrics = QFontMetrics(widget.font())
    width = font_metrics.horizontalAdvance('M') * max_chars
    widget.setMaximumWidth(width)

def update_napari_layer_combos(viewer, combos_dict, merge_combo=None, merge_button=None, on_change_callback=None):
    """
    Update napari layer comboboxes when layers are added/removed/renamed.
    
    This utility function manages the common pattern of syncing QComboBox widgets
    with napari's layer list, while handling signal connections to avoid duplicates.
    
    Args:
        viewer: napari.viewer.Viewer instance
        combos_dict: Dict mapping combo names to (QComboBox, layer_class_name) tuples.
                    e.g., {'image': (self.active_image, 'Image'), 
                           'points': (self.active_points, 'Points')}
        merge_combo: Optional QComboBox for a "merge labels" style selection
        merge_button: Optional QPushButton to enable/disable based on merge_combo content
        on_change_callback: Optional callback function to call after updating combos
    
    Example:
        update_napari_layer_combos(
            self.viewer,
            {
                'image': (self.active_image, 'Image'),
                'points': (self.active_points, 'Points'),
                'label': (self.active_label, 'Labels')
            },
            merge_combo=self.active_merge_label,
            merge_button=self.mlsbtn,
            on_change_callback=self.update_possible_functions
        )
    """
    # Collect layers by type
    layer_dict = {}
    for combo_name, (combo, layer_class) in combos_dict.items():
        layer_dict[combo_name] = [l.name for l in viewer.layers if l.__class__.__name__ == layer_class]
    
    # Save current selections
    current_selections = {combo_name: combos_dict[combo_name][0].currentText() 
                         for combo_name in combos_dict}
    if merge_combo:
        merge_selection = merge_combo.currentText()
    
    # Update merge button state and populate merge_combo if provided
    if merge_combo and merge_button:
        merge_combo.clear()
        merge_combo.addItems(layer_dict.get('label', []))
        if len(layer_dict.get('label', [])) <= 1:
            merge_combo.setEnabled(False)
            merge_button.setEnabled(False)
        else:
            merge_combo.setEnabled(True)
            merge_button.setEnabled(True)
    
    # Clear and repopulate combos
    for combo_name, (combo, _) in combos_dict.items():
        combo.clear()
        combo.addItems(layer_dict[combo_name])
    
    # Restore selections if available
    for combo_name, selection in current_selections.items():
        combo = combos_dict[combo_name][0]
        if selection in layer_dict[combo_name]:
            combo.setCurrentText(selection)
    
    if merge_combo and merge_selection in layer_dict.get('label', []):
        merge_combo.setCurrentText(merge_selection)
    
    # Reconnect layer name change events (disconnect first to avoid duplicates)
    if on_change_callback is not None:
        for layer in viewer.layers:
            try:
                layer.events.name.disconnect(on_change_callback)
            except (TypeError, RuntimeError, ValueError):
                pass
            layer.events.name.connect(on_change_callback)


def create_resolution_group(widget_instance):#, xy_res_changed_callback=None, z_res_changed_callback=None):
    def _update_xy_res(val):
        widget_instance.xyres = val
        widget_instance.xyresbox.setValue(widget_instance.xyres)
    
    def _update_z_res(val):
        widget_instance.zres = val
        widget_instance.zresbox.setValue(widget_instance.zres)
    
    def _update_z_scale(state):
        widget_instance.z_scale_state = state
        widget_instance.z_scale.setChecked(state)

    if widget_instance.parent_widget:
        widget_instance.parent_widget.xy_res_changed.connect(_update_xy_res)
        widget_instance.parent_widget.z_res_changed.connect(_update_z_res)
        widget_instance.parent_widget.z_scale_state_changed.connect(_update_z_scale)
    
    def _on_xy_res_change(value):
        widget_instance.xyres = value
        _set_z_scale(widget_instance.z_scale.checkState()) 
        if widget_instance.parent_widget:
            widget_instance.parent_widget.xy_res_changed.emit(value)

    def _on_z_res_change(value):
        widget_instance.zres = value
        _set_z_scale(widget_instance.z_scale.checkState()) 
        if widget_instance.parent_widget:
            widget_instance.parent_widget.z_res_changed.emit(value)
    
    def _on_z_scale_change(state):
        widget_instance.z_scale_state = state
        if widget_instance.parent_widget:
            widget_instance.parent_widget.z_scale_state_changed.emit(state) 

    def _set_z_scale(state):
        layers = widget_instance.viewer.layers
        if state == 2:  # Checked
            try:
                layers.events.inserted.disconnect(scale_layers)
            except (TypeError, RuntimeError):
                pass
            layers.events.inserted.connect(scale_layers)
            try:
                layers.events.removed.disconnect(scale_layers)
            except (TypeError, RuntimeError):
                pass
            layers.events.removed.connect(scale_layers)
            scale_layers()
        else:
            try:
                layers.events.inserted.disconnect(scale_layers)
                layers.events.removed.disconnect(scale_layers)
            except (TypeError, RuntimeError):
                pass
            for layer in layers:
                layer.scale = [1, 1, 1]

    def scale_layers(event=None):
        for layer in widget_instance.viewer.layers:
            if widget_instance.xyresbox.value() == 0 or widget_instance.zresbox.value() == 0:
                return
            z_scale_factor = widget_instance.zresbox.value() / widget_instance.xyresbox.value()
            layer.scale = [z_scale_factor, 1, 1]

    # Create spin boxes
    xyresbox = QDoubleSpinBox()
    zresbox = QDoubleSpinBox()
    z_scale = QCheckBox("Scale z dimension")
    z_scale.stateChanged.connect(_set_z_scale)
    z_scale.stateChanged.connect(_on_z_scale_change)
    
    # Setup spin boxes with default parameters
    _setup_spin(
        widget_instance, xyresbox,
        minval=0, val=1, step=0.05, attrname='xyres',
        dec=4, dtype=float, connect_func=_on_xy_res_change
    )
    _setup_spin(
        widget_instance, zresbox,
        minval=0, val=1, step=0.05, attrname='zres',
        dec=4, dtype=float, connect_func=_on_z_res_change
    )
    
    # Create layout
    res_box = QHBoxLayout()
    res_box.addWidget(QLabel('xy (um/px):'))
    res_box.addWidget(xyresbox)
    res_box.addWidget(QLabel('z (um/px):'))
    res_box.addWidget(zresbox)

    res_box2 = QVBoxLayout()
    res_box2.addLayout(res_box)
    res_box2.addWidget(z_scale)
    
    # Create group box
    res_group = QGroupBox('Resolution')
    res_group.setLayout(res_box2)
    
    return res_group, xyresbox, zresbox, z_scale

