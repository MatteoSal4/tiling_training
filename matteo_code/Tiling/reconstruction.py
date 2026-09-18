"""
reconstruction.py
"""
import numpy as np
import torch
import torch.nn.functional as F


def trilinear_upsample(volume, scale_factor=2):
    """
    Upscale a 3D volume with trilinear interpolation (PyTorch).

    Parameters
    ----------
    volume       : (Z, Y, X) ndarray
    scale_factor : int — upscaling factor applied to all 3 axes (default 2)

    Returns
    -------
    upsampled : (Z*scale, Y*scale, X*scale) ndarray, same dtype as input
    """
    original_dtype = volume.dtype

    # PyTorch requires shape (batch, channels, Z, Y, X) → add 2 dummy dimensions
    tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)  # (1, 1, Z, Y, X)

    # trilinear interpolation: align_corners=False keeps boundary voxels aligned
    upsampled = F.interpolate(tensor, scale_factor=scale_factor, mode='trilinear', align_corners=False)

    # remove dummy dimensions and convert back to numpy with the original dtype
    return upsampled.squeeze(0).squeeze(0).numpy().astype(original_dtype)


def reconstruct_volume_from_tiles(tiles, tile_shape, volume_shape):
    """
    Reconstruct a volume by placing tiles at their original coordinates.
    Voxels covered by multiple tiles are averaged. Uncovered voxels remain zero.

    Parameters
    ----------
    tiles        : list[dict] — must contain 'tile_dose' and 'origin'
    tile_shape    : (tz, ty, tx)
    volume_shape : (Z, Y, X)

    Returns
    -------
    volume : (Z, Y, X) float32 ndarray
    """
    # accumulator sums the dose of all tiles covering each voxel
    # count tracks how many tiles cover each voxel
    accumulator = np.zeros(volume_shape, dtype=np.float64)
    count       = np.zeros(volume_shape, dtype=np.int32)

    for t in tiles:
        # top-left corner of the tile in the volume (can be negative if it goes outside the border)
        start_z = t['origin'][0]
        start_y = t['origin'][1]
        start_x = t['origin'][2]
        size_z  = tile_shape[0]
        size_y  = tile_shape[1]
        size_x  = tile_shape[2]

        # coordinates of the region in the volume (clamped to borders)
        vol_z0 = max(0, start_z)
        vol_z1 = min(volume_shape[0], start_z + size_z)
        vol_y0 = max(0, start_y)
        vol_y1 = min(volume_shape[1], start_y + size_y)
        vol_x0 = max(0, start_x)
        vol_x1 = min(volume_shape[2], start_x + size_x)

        # offset inside the tile (>0 if the tile starts outside the volume on the left)
        tile_z0 = vol_z0 - start_z
        tile_y0 = vol_y0 - start_y
        tile_x0 = vol_x0 - start_x

        # slices selecting the corresponding region in the volume and in the tile
        in_vol  = np.s_[vol_z0:vol_z1, vol_y0:vol_y1, vol_x0:vol_x1]
        in_tile = np.s_[tile_z0:tile_z0+(vol_z1-vol_z0),
                        tile_y0:tile_y0+(vol_y1-vol_y0),
                        tile_x0:tile_x0+(vol_x1-vol_x0)]

        # add this tile's dose to the accumulator and increment the counter
        accumulator[in_vol] += t['tile_dose'][in_tile]
        count[in_vol]       += 1

    # divide by the number of tiles covering each voxel (average)
    # where count == 0 (no tile) the voxel stays zero
    # count_safe avoids division by zero (np.where evaluates both branches)
    count_safe = np.where(count > 0, count, 1)
    volume     = np.where(count > 0, accumulator / count_safe, 0.0).astype(np.float32)
    return volume
