from scipy.ndimage import gaussian_filter, distance_transform_edt, center_of_mass
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.segmentation import watershed


def get_slices(rad_xy, rad_z, loc, shape):
    x1 = max(loc[2] - rad_xy, 0) ;
    x2 = min(loc[2] + rad_xy, shape[2]) ;
    y1 = max(loc[1] - rad_xy, 0) ;
    y2 = min(loc[1] + rad_xy, shape[1]) ;
    z1 = max(loc[0] - rad_z, 0) ;
    z2 = min(loc[0] + rad_z, shape[0]) ;
    relx = loc[2] - x1 ;
    rely = loc[1] - y1 ;
    relz = loc[0] - z1 ;

    return slice(z1,z2), slice(y1,y2), slice(x1,x2), [relz, rely, relx]

def dist_watershed_sep(mask, loc):
    dists = distance_transform_edt(mask, sampling=[4,1,1])

    pk_idx = peak_local_max(dists, labels=mask)
    pks = np.zeros_like(dists, dtype=bool)
    pks[pk_idx] = True
    pk_labels = label(pks)
    if pk_labels.max()>1:
        merged_peaks = center_of_mass(pks, pk_labels, index=range(1, pk_labels.max()+1))
        merged_peaks = np.round(merged_peaks).astype('int')

        markers = np.zeros_like(mask, dtype='int')
        for i in range(merged_peaks.shape[0]):
            markers[merged_peaks[i,0], merged_peaks[i,1], merged_peaks[i,2]] = i+1

        labels = watershed(-dists, markers=markers, mask=mask)
        wantLabel = labels[loc[0], loc[1], loc[2]]
        mask_out = labels == wantLabel

    else:
        mask_out = mask

    return mask_out

def pt_to_seg(
        img, 
        pts,
        rad_xy=6, 
        rad_z=4, 
        snap_rad=2, 
        max_rad_xy=2, 
        max_rad_z=1, 
        blur_sig=[0.5,0.7,0.7], 
        snap_to_max=True
        ):

    #img = viewer.layers[0].data
    #labels = viewer.layers[1].data
    #curr_n = labels.max()
    #pts = viewer.layers[2].data

    x = pts[:,2]
    y = pts[:,1]
    z = pts[:,0]
    n = len(z)

    img_inv = gaussian_filter(np.min(img) + np.max(img) - img, blur_sig)

    w = img.shape[2]
    h = img.shape[1]
    d = img.shape[0]

    # make markers:
    markers = np.zeros_like(img, dtype='int')

    for j in range(n):
        pos = np.round([z[j], y[j], x[j]]).astype('int')

        if snap_to_max:
            zrange, yrange, xrange, rel_pos = get_slices(snap_rad, snap_rad, pos, img.shape)
            pointIntensity = img_inv[zrange, yrange, xrange]

            shift = np.unravel_index(np.argmin(pointIntensity), pointIntensity.shape)
            shift = np.asarray(shift)-snap_rad

            z[j] = z[j] + shift[0]
            y[j] = y[j] + shift[1]
            x[j] = x[j] + shift[2]

            pos = np.round([z[j], y[j], x[j]]).astype('int')

        markers[pos[0], pos[1], pos[2]] = j+1#+curr_n

    # make mask:
    mask = np.zeros_like(img, dtype='bool')

    for j in range(n):
        pos = np.round([z[j], y[j], x[j]]).astype('int')
        pointIntensity = img_inv[pos[0], pos[1], pos[2]]

        # find local min (inverted max) value:
        zrange, yrange, xrange, rel_pos = get_slices(max_rad_xy, max_rad_z, pos, img.shape)
        subim = img_inv[zrange, yrange, xrange]
        local_min = np.min(subim)
        # get local region to threshold, find local min value:
        zrange, yrange, xrange, rel_pos = get_slices(rad_xy, rad_z, pos, img.shape)
        subim = img_inv[zrange, yrange, xrange]
        local_max = np.max(subim) # background
        #tifffile.imwrite(os.path.join(label_dir,"subim_"+str(j)+".tif"), np.min(subim, axis=0))

        # threshold:
        thresh = 0.5*local_min + 0.5*local_max
        if thresh < pointIntensity:
            print("threshold overriden for spot "+str(j)+" "+str(thresh)+" "+str(pointIntensity))
            thresh = 0.5*local_max + 0.5*pointIntensity
        subim_mask = subim <= thresh

        # check for multiple objects:
        sublabels = label(subim_mask)
        if sublabels.max() > 1:
            wantLabel = sublabels[rel_pos[0], rel_pos[1], rel_pos[2]]
            subim_mask = sublabels == wantLabel

            # recheck max:
            thresh2 = 0.5*np.min(subim[subim_mask]) + 0.5*np.max(subim)
            if thresh < thresh2:
                subim_mask = subim <= thresh2
                sublabels = label(subim_mask)
                wantLabel = sublabels[rel_pos[0], rel_pos[1], rel_pos[2]]
                subim_mask = sublabels == wantLabel

        pt_solidity = regionprops(subim_mask.astype('int'))[0].solidity

        if pt_solidity < 0.8:
            subim_mask = dist_watershed_sep(subim_mask, rel_pos)
        #tifffile.imwrite(os.path.join(label_dir,"subim_mask_"+str(j)+".tif"), np.max(subim_mask, axis=0))
        submask = mask[zrange, yrange, xrange]
        submask = np.logical_or(submask, subim_mask)

        mask[zrange, yrange, xrange] = submask

    outlabels = watershed(img_inv, markers=markers, mask=mask)

    return outlabels


