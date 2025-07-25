from qtpy.QtGui import QFontMetrics

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
