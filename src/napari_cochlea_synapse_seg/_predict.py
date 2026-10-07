import glob
import gunpowder as gp
import numpy as np
import os
import torch
from skimage.measure import label, regionprops_table
from skimage.feature import peak_local_max

def predict(
    checkpoint,
    raw_file,
    raw_dataset):

    try:
        from funlib.learn.torch.models import UNet, ConvPass
    except ImportError as exc:
        raise ImportError(
            "Prediction requires 'funlib.learn.torch', which is not available on PyPI. "
            "Install it manually with: pip install git+https://github.com/funkelab/funlib.learn.torch.git"
        ) from exc

    raw = gp.ArrayKey('RAW')
    pred = gp.ArrayKey('PRED')

    voxel_size = gp.Coordinate((4,1,1))

    input_shape = gp.Coordinate((44,172,172))
    output_shape = gp.Coordinate((24,80,80))

    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size
 
    context = (input_size - output_size) / 2

    scan_request = gp.BatchRequest()

    scan_request.add(raw, input_size)
    scan_request.add(pred, output_size)

    in_channels = 1
    num_fmaps = 12
    fmap_inc_factor = 5
    
    downsample_factors = [(1,2,2),(1,2,2),(2,2,2)]

    kernel_size_down = [
                [(3,)*3, (3,)*3],
                [(3,)*3, (3,)*3],
                [(3,)*3, (3,)*3],
                [(1,3,3), (1,3,3)]]

    kernel_size_up = [
                [(1,3,3), (1,3,3)],
                [(3,)*3, (3,)*3],
                [(3,)*3, (3,)*3]]

    unet = UNet(
        in_channels=in_channels,
        num_fmaps=num_fmaps,
        fmap_inc_factor=fmap_inc_factor,
        downsample_factors=downsample_factors,
        kernel_size_down=kernel_size_down,
        kernel_size_up=kernel_size_up,
        constant_upsample=True)
    
    model = torch.nn.Sequential(
            unet,
            ConvPass(num_fmaps, 1, [[1,]*3], activation='Tanh'))
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
 
    source = gp.ZarrSource(
        raw_file,
            {
                raw: raw_dataset
            },
            {
                raw: gp.ArraySpec(
                    interpolatable=True,
                    voxel_size=voxel_size)
            })
    with gp.build(source):
        raw_roi = source.spec[raw].roi

    # Volumes smaller than one output window (e.g. few z-slices) need extra
    # padding so Scan has a full input/output chunk to fit; cropped afterwards.
    raw_vox = np.array(raw_roi.get_shape()) / np.array(voxel_size)
    out_vox = np.array(output_shape)
    extra_vox = np.maximum(0, np.ceil((out_vox - raw_vox) / 2)).astype(int)
    extra = gp.Coordinate(tuple(int(e) for e in extra_vox)) * voxel_size

    source += gp.Pad(raw, context + extra)
    source += gp.Normalize(raw)
    source += gp.Unsqueeze([raw])

    with gp.build(source):
        total_input_roi = source.spec[raw].roi
        total_output_roi = total_input_roi.grow(-context, -context)

    model.eval()

    predict = gp.torch.Predict(
        model=model,
        checkpoint=checkpoint,
        inputs = {
            'input': raw
        },
        outputs = {
            0: pred
        })

    scan = gp.Scan(scan_request)

    pipeline = source
    # d,h,w

    pipeline += gp.Stack(1)
    # b,d,h,w

    pipeline += predict
    pipeline += scan
    
    pipeline += gp.Squeeze([raw, pred])
    pipeline += gp.Squeeze([raw, pred])

    predict_request = gp.BatchRequest()

    predict_request.add(raw, total_input_roi.get_end())
    predict_request[raw].roi = total_input_roi
    
    predict_request.add(pred, total_output_roi.get_end())
    predict_request[pred].roi = total_output_roi

    with gp.build(pipeline):
        batch = pipeline.request_batch(predict_request)

    data = batch[pred].data
    if extra_vox.any():
        data = data[tuple(slice(int(e), data.shape[i] - int(e)) for i, e in enumerate(extra_vox))]
    return data