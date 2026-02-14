import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import regionprops, regionprops_table, label
from skimage.feature import peak_local_max
from skimage.filters import threshold_yen, threshold_otsu
from skimage.segmentation import watershed
from skimage.util import map_array

def calc_errors(labels, gt_xyz):

    gtp = gt_xyz.shape[0]
    Npoints_pred = len(np.unique(labels))-1

    pred_idcs = np.zeros(gtp).astype(np.int32)

    for i in range(gtp):

        cent = np.flip(np.around(gt_xyz[i,:]).astype(np.int16))
        pred_idcs[i] = labels[tuple(cent)]

    idc, counts = np.unique(pred_idcs[pred_idcs>0], return_counts=True)

    tp = len(idc) #np.sum(counts==1)
    fm = np.sum(counts-1)
    fn = np.sum(pred_idcs==0)
    fp = Npoints_pred - len(idc)   
    
    ap = tp/(tp+fn+fp+fm)
    f1 = 2*tp/(2*tp + fp + fn + fm)
    if tp+fp == 0:
        prec = 1
    else:
        prec = tp/(tp+fp)
    rec  = tp/gtp

    results = {
        "gtp": gtp, # ground truth positives
        "tp": tp,   # true positives
        "fn": fn,   # false negatives 
        "fm": fm,   # false merges (2 ground truth positives IDed as 1 spot)
        "fp": fp,   # false potives
        "ap": ap,   # average precision
        "f1": f1,   # f1 score 
        "precision": prec, # precision
        "recall": rec,     # recall
        "mergerate": fm/gtp, # merge rate 
    }
    
    return results 

def sdt_to_labels(pred, 
                  peak_thresh=0.2,
                  mask_thresh=0.0,
                  strict_peak_thresh=True,
                  blur_sig=[0.5, 0.7, 0.7],
                  size_filt=None):
    dist_map = gaussian_filter(pred, blur_sig) 
    coords = peak_local_max(
            dist_map, 
            footprint=np.ones((3, 3, 3)), 
            threshold_abs=peak_thresh, 
            min_distance=2,
            )
    markers = np.zeros(dist_map.shape, dtype=bool)
    markers[tuple(coords.T)] = True
    markers = label(markers)

    mask = pred>(mask_thresh)

    markers = markers*mask
    segmentation = watershed(-dist_map, markers, mask=mask)
    
    if strict_peak_thresh: # remove any masked regions without a peak  over peak threshold
         strict_labels = np.unique(segmentation[tuple(coords.T)])
         for i in np.unique(segmentation):
             if i == 0:
                 continue
             if i not in strict_labels:
                 segmentation[segmentation==i] = 0
    
    segmentation = filt_labels_by_size(segmentation, size_filt=size_filt)
    return segmentation

def filt_labels_by_size(segmentation, 
                        size_filt=1):
    if size_filt is not None and size_filt>1:
        seg_props = regionprops_table(segmentation, properties=('label', 'num_pixels'))
        in_labels = seg_props['label']
        out_labels = in_labels
        out_labels[seg_props['num_pixels']<size_filt] = 0

        segmentation_filt = map_array(segmentation, in_labels, out_labels)
        return segmentation_filt
    else:
        return segmentation

def bksub_threshold(img, type='yen'):
    img = img.astype(np.float32)
    img_filt = gaussian_filter(img, sigma=(1,2,2)) - gaussian_filter(img, sigma=(2,10,10))
    if type=='otsu':
        img_thresh = threshold_otsu(np.max(img_filt, axis=0))
    elif type=='yen':
        img_thresh = threshold_yen(np.max(img_filt, axis=0))
    img_map = img_filt > img_thresh 
    labels = watershed(-img_filt, mask=img_map)

    return labels
