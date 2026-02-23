from qtpy.QtGui import QFontMetrics

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
        spinbox.valueChanged[dtype].connect(lambda value: setattr(curr_class, attrname, value))
        setattr(curr_class, attrname, spinbox.value())

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
