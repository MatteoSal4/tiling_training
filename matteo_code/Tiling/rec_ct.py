import numpy as np
from beam_tiling import _extract_tile
from reconstruction import trilinear_upsample, reconstruct_volume_from_tiles


def reconstruct_ct_from_centers(ct_volume, centers, center_shape, scale_factor=2):
    """
    Reconstruct the CT volume using tiles centered at the provided positions.
    Everything not covered by a tile is set to 0.

    The CT is upsampled by scale_factor× so that center coordinates
    (already in upsampled dose space) work directly.

    Parameters
    ----------
    ct_volume    : (Z, Y, X) ndarray — original CT volume
    centers      : (N, 3) array — center coordinates in upsampled dose space
    center_shape : (cz, cy, cx) — tile dimensions to extract
    scale_factor : int — upsampling factor (default 2, same as dose)

    Returns
    -------
    ct_rec : (Z, Y, X) ndarray — reconstructed upsampled CT (0 outside tiles)
    """
    ct_up = trilinear_upsample(ct_volume, scale_factor=scale_factor)  # bring CT into the same space as the dose

    dimensioni_volume = np.array(ct_up.shape)
    dimensioni_centro = np.array(center_shape)

    tiles_estratti = []
    for centro in centers:
        # skip centers whose tile lies completely outside the volume
        inizio = np.round(np.array(centro) - dimensioni_centro / 2).astype(int)
        fine   = inizio + dimensioni_centro
        if np.any(fine <= 0) or np.any(inizio >= dimensioni_volume):
            continue

        tile_ct, origine = _extract_tile(ct_up, centro, center_shape)
        tiles_estratti.append({'tile_dose': tile_ct, 'origin': origine})

    return reconstruct_volume_from_tiles(tiles_estratti, center_shape, ct_up.shape)
