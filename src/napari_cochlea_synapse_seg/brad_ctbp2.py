import napari
import numpy as np
import scipy as sp
from magicgui.widgets import CheckBox, Container, PushButton, create_widget
from napari.layers import Image, Points
from skimage.draw import polygon2mask
from skimage.feature import blob_log
from skimage.util import img_as_float


class CtBP2Detection(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        self._viewer.layers.events.inserted.connect(
            self._rescan_layers, position="last"
        )
        self._viewer.layers.events.removed.connect(
            self._rescan_layers, position="last"
        )

    def _rescan_layers(self):
        self._roi_map = {}
        for layer in self._viewer.layers:
            if isinstance(layer, Points):
                layer.mouse_drag_callbacks.append(self._mouse_click)

    def _mouse_click(self, layer, event):
        if layer.mode != "pan_zoom":
            # A tool (e.g., add, remove, select) is being used. Don't interfere
            # with what's already going on otherwise we may end up with
            # duplicate points or delete two points.
            return
        if event.buttons[0] == 2:
            if "Shift" in event.modifiers:
                self._remove_point(layer, event)
            else:
                self._add_point(layer, event)

    def _remove_point(self, layer, event):
        try:
            position = layer.world_to_data(event.position)
            
            if len(layer.data) == 0:
                return
                
            distances = np.linalg.norm(layer.data - position, axis=1)
            closest_idx = np.argmin(distances)
            closest_distance = distances[closest_idx]
            
            new_data = np.delete(layer.data, closest_idx, axis=0)
            layer.data = new_data
            print(f"Removed point at index {closest_idx}")
                
        except Exception as e:
            print(f"Error removing point: {e}")
            import traceback
            traceback.print_exc()

    def _add_point(self, layer, event):
        image_layer = self._image_layer_combo.value
        if image_layer is None:
            print("No image layer selected")
            return
        
        print(f"Adding point. View mode: {'2D' if self._viewer.dims.ndisplay == 2 else '3D'}")
        
        if self._viewer.dims.ndisplay == 2:
            # Logic for handling 2D view.
            near_point = list(event.position)
            far_point = list(event.position)
    
            # Find the axis to project the ray along.
            ray_axis = ({0, 1, 2} - set(self._viewer.dims.displayed)).pop()
    
            # Get the thickness of the view. The thickness is the full range
            # (lower to upper), but point click is in the center.
            thickness = self._viewer.dims.thickness[ray_axis] / 2
            near_point[ray_axis] += thickness
            far_point[ray_axis] -= thickness
            near_point = image_layer.world_to_data(near_point)
            far_point = image_layer.world_to_data(far_point)
            
            print(f"2D mode - Near: {near_point}, Far: {far_point}, Thickness: {thickness}")
            
            if thickness == 0:
                print("Zero thickness, adding point directly at near_point")
                layer.add(near_point)
                return
        else:
            # Logic for handling 3D view.
            # Find coordinates where ray enters/exists layer bounding box.
            near_point, far_point = image_layer.get_ray_intersections(
                event.position, event.view_direction, event.dims_displayed
            )
            if (near_point is None) or (far_point is None):
                print("Ray intersection failed - no valid near/far points")
                return
            
            print(f"3D mode - Near: {near_point}, Far: {far_point}")
    
        try:
            num_samples = 25
            ray = np.linspace(near_point, far_point, num_samples, endpoint=True)
            
            ray_clamped = np.copy(ray)
            for i in range(ray.shape[1]):
                ray_clamped[:, i] = np.clip(ray[:, i], 0, image_layer.data.shape[i] - 1)
            
            intensities = sp.ndimage.map_coordinates(
                image_layer.data,
                ray_clamped.T,
                order=1,  
                mode="constant",
                cval=0,
                prefilter=False,  
            )
            
            max_idx = intensities.argmax()
            max_point = ray[max_idx]
            print(f"Max intensity: {intensities[max_idx]} at point: {max_point}")
            
            layer.add(max_point)
            print(f"Point added successfully")
            
        except Exception as e:
            print(f"Error in _add_point: {e}")
            import traceback
            traceback.print_exc()