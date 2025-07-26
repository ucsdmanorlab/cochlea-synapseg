# Cochlea-SynapSeg

[![License BSD-3](https://img.shields.io/pypi/l/cochlea-synapseg.svg?color=green)](https://github.com/ucsdmanorlab/cochlea-synapseg/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/cochlea-synapseg.svg?color=green)](https://pypi.org/project/cochlea-synapseg)
[![Python Version](https://img.shields.io/pypi/pyversions/cochlea-synapseg.svg?color=green)](https://python.org)
[![DOI](https://zenodo.org/badge/865642960.svg)](https://doi.org/10.5281/zenodo.16433552)
<!--
[![tests](https://github.com/ucsdmanorlab/cochlea-synapseg/workflows/tests/badge.svg)](https://github.com/ucsdmanorlab/cochlea-synapseg/actions)
[![codecov](https://codecov.io/gh/ucsdmanorlab/cochlea-synapseg/branch/main/graph/badge.svg)](https://codecov.io/gh/ucsdmanorlab/cochlea-synapseg)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/cochlea-synapseg)](https://napari-hub.org/plugins/cochlea-synapseg)
-->

A napari plugin to segment cochlear ribbon synapses. 

More is in the works, but currently includes:
(1) pre-processing functions, 
(2) tools to quickly generate ground truth ribbon segmentation, 
(3) deep-learning based ribbon segmentation prediction, and 
(4) tools to check for synapse pairs, and export montage images.

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `cochlea-synapseg` (recommended: in a new conda environment with up-to-date napari) via [pip]:

    python -m pip install cochlea-synapseg

## Usage

After installation, you can find the plugin next time you launch napari (_Plugins > Cochlea SynapSeg_ > SynapSeg Widget / Montage Widget).

SynapSeg Widget includes all core functionalities. SynapSeg widget is divided into multiple tabs and sections, for "quick use", be sure to check the settings denoted with asterisks below.

Montage Widget is under development and will later be rolled into the SynapSeg Widget. 

Jump to: [Preprocess](#preprocess-tab) | [Ground Truth](#ground-truth-tab) | [Predict](#predict-tab) | [Montage Widget](#montage-widget)

----------------------------------
### Preprocess Tab
<img width="268" height="323" alt="preprocess" src="https://github.com/user-attachments/assets/2229158a-9650-49e0-86b4-2d3f53b656f2" />

#### Image tools
* **image layer** - select an image layer (must already be loaded into napari) for preprocessing
* **xy/z resolution** - (optional) in um/pixel, auto-loaded from .tifs when available in metadata
* **split channels** - if your loaded image layer contains multiple channels, select the channel axis and click split channels. Channels must be separated for ribbon segmentation. (defaults to the smallest dimension)
#### Points tools
* **points layer** - select a points layer (must already be loaded into napari) to use the preprocessing tools below:
* **real -> pixel units** - if you've loaded points that were saved in real units, make sure the pixel size information above in image tools is correct, then click "real -> pixel units" to convert
* **chan ->z convert** - some points (like ImageJ/FIJI rois or CellCounter points), show up in the wrong z plane because their "slice" coordinates are a combination of both slice and channel info. If this happens, set the number of channels (in the original image, where the ROIs were created!), and then click "chan -> z convert". Z coordinates of the points layer will be divided by the number of channels specified.

Jump to: [Usage](#usage) | [Preprocess](#preprocess-tab) | [Ground Truth](#ground-truth-tab) | [Predict](#predict-tab) | [Montage Widget](#montage-widget)

----------------------------------
### Ground Truth Tab
#### Image Tools
<img width="331" height="123" alt="GT_image" src="https://github.com/user-attachments/assets/c797d423-d7f5-43e4-8940-a87c235b9c2d" />


* \***presynaptic layer** - use the dropdown to select a loaded image layer that contains the ribbon stain
* **xy/z resolution** - (optional) in um/pixel, auto-loaded from .tifs when available in metadata, used for scaling
* **Scale z dimension** - check to scale all layers in the 3D viewer for isotropic viewing

#### Points Tools & Points to Labels
<img width="331" height="363" alt="GT_pts" src="https://github.com/user-attachments/assets/04bd8b38-64ed-4b91-a1b2-f002a7788d40" />

**1. Points Layer Selection** - use the dropdown to select an existing points layer (or skip to #8 if not loading in existing points)

**2. Find peaks above** - use an automatic peak finder to find peaks above a certain value. Useful as a starting point for manual annotation. 

**3. Guess** - use the image intensity information to guess an appropriate peak value for #2

\***4. New Points Layer** - if starting annotation from scratch, click to create a new points layer

\***5. Rotate to XY and \*6. Auto-adjust Z** - these convenient functions allow you to quickly annotate points in Napari's 3D view. Click "Rotate to XY" before adding new points. These points will now have the correct XY position but will have missing Z information. (Rotate out of XY to confirm.) Click "Auto-adjust z" and the z will automatically adjust to the brightest point. 

**7. Manually Edit Z** - useful for overlapping points, can be used to manually edit the z position of ONLY selected points. Use the +/- arrow keys for single z steps type in a number to move a larger amount. 

**8. Snap too Max** - will automatically adjust all points to their local max (search radius defined in pixels in Advanced Settings -> snap to max rad). Useful for adjusting quickly dropped points, but proceed with caution if you have close-together points. 

\***9. Points to Labels** - the key functionality of the module, creates a label layer by performing a local segmentation on all points.

**10. Advanced Settings** - adjust settings for the points to labels function to optimize local segmentation and watershed separation of points. 

### Labels Tools
<img width="332" height="291" alt="GT_labels" src="https://github.com/user-attachments/assets/e6dadc17-9df4-4f45-913a-44ed348b09d2" />

**1. Labels Layer Selection** - use the dropdown to select the labels layer that represents ribbon segmentation 

**2. Make Labels Editable** - some file formats (including zarrs) load in as dask arrays, which don't allow editing. Checking this box will make the labels layer editable to add/remove new labels by converting to a numpy array (will load the layer into memory, so be careful if dealing with large images!).

**3. New label** - if hand-painting labels, set the active label in the label layer to an unused ID before painting

\***4. Remove a Label** - use the labels layer eyedropper tool to identify the ID of an unwanted label, then type in the box and click "Remove label"

\***5. Merge labels** - if iteratively creating new labels, merge existing labels (in dropdown #1) with new labels (specified in dropdown #5). This function will automatically ensure overlapping label IDs are not used. 

**6. Labels to Points** - an existing label layer can be converted to a points layer based on label centroids. Used with #7. 

**7. Keep labels from points** - After using (#6), use the points editing tools to quickly remove unwanted points. Click (#7) to retain only the labels that correspond with a point. (Labels specified in "labels layer", points specified in "points layer" above.)

### Save to Zarr

![save_zarr](https://github.com/user-attachments/assets/1d824f49-012f-4fac-8fa1-64d7d319cd34)

Functionality to save to .zarr format. Saves presynapse image as 'raw', and labels as 'labeled' if they exist. Used for later prediction of ribbon segmentation.  

\***21. File Path** - the directory in which to save the zarr; use the folder icon to search for an existing directory

\***22. File Name** - the zarr name to save to; use the magnifying glass icon to select an existing .zarr

**23. From Source** - set the file path and name to where the image layer was loaded from. (Caution: if you loaded a zarr, this will result in the zarr being overwritten!)

\***24. Save zarr** - saves the presynapse image layer (as selected above), and labels layers (as selected, if it exists) in the specified .zarr, as 'raw' and 'labeled', respectively. These can be drag + dropped into napari for viewing later, and can be fed directly into prediction. 

Jump to: [Usage](#usage) | [Preprocess](#preprocess-tab) | [Ground Truth](#ground-truth-tab) | [Predict](#predict-tab) | [Montage Widget](#montage-widget)

----------------------------------
### Predict Tab

#### Predict
<img width="336" height="291" alt="image" src="https://github.com/user-attachments/assets/1c83529d-2102-433b-999e-5bc75c884252" />

* **model path** - location of pre-trained model (defaults to saved ribbon model in this repo)
* **input zarr path** - select a zarr (saved using the tools in Ground Truth -> Save Zarr) on which to predict
* **predict** - run prediction (may be slow if GPU is not enabled), output is saved in zarr
* **show** - show prediction as a new layer in the napari viewer. Prediction is a distance transform (values between -1 and 1), and can be converted to labels using the "labels from prediction" functions below.

#### Labels from prediction
<img width="337" height="217" alt="image" src="https://github.com/user-attachments/assets/d63ba53c-e617-418a-8ea8-083cbaf56d13" />

* **pred layer** - select the prediction layer loaded in the viewer
* **mask threshold** - the threshold used to determine the _object boundaries_ from the prediction. For most images, 0-0.1 works well.
* **peak threshold** - the threshold used to determine whether nearby objects will be split, and whether objects without a bright center will be retained. For most images, 0.1-0.4 works well.
* **Prediction to labels** - generate labels using the settings selected above

Jump to: [Usage](#usage) | [Preprocess](#preprocess-tab) | [Ground Truth](#ground-truth-tab) | [Predict](#predict-tab) | [Montage Widget](#montage-widget)

----------------------------------
### Montage Widget
<img width="445" height="447" alt="montage-paired" src="https://github.com/user-attachments/assets/905d0a98-b9b0-4c00-9bcf-1eb947c1d4b4" />

Used to generate montages of synapses and orphan ribbons (red), and tools to quickly navigate between montage view and the original image. Documentation coming soon. 

<!-- 
## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.
-->
## License

Distributed under the terms of the [BSD-3] license,
"cochlea-synapseg" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[file an issue]: https://github.com/ucsdmanorlab/cochlea-synapseg/issues/new

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
