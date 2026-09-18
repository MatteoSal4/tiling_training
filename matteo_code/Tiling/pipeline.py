import numpy as np
from center_reconstruction import setup_beam, lateral_expansion, axial_sliding, reconstruct_from_centers, residual_grabber


def run_pipeline(dose, tile_shape, center_shape, threshold_fraction, radius, step):
    """
    Complete pipeline: from dose volume to final center coordinates.

    Parameters
    ----------
    dose               : (Z, Y, X) ndarray — original dose volume
    tile_shape         : (tz, ty, tx) — size of the large tile used for lateral expansion
    center_shape       : (cz, cy, cx) — size of the central region used for reconstruction
    threshold_fraction : float — dose threshold as a fraction of the maximum (e.g. 0.1 = 10%)
    radius             : float — beam mask cylinder radius (voxels, in upsampled space)
    step               : float — distance between successive centers along the beam axis (voxels)

    Returns
    -------
    centri_finali : (N, 3) ndarray — global coordinates (Z, Y, X) of all centers
    """
    # setup: upsample, bragg peak, entry point, direction, beam mask
    dose_up, bragg, entry, direction, mask = setup_beam(dose, threshold_fraction, radius)

    dose_threshold = dose_up.max() * threshold_fraction  # absolute dose threshold

    # phase 1: find the lateral pattern at the entry point
    center_list = lateral_expansion(dose_up, entry, direction, tile_shape, center_shape, dose_threshold)

    # phase 2: slide the pattern along the beam from entry to Bragg peak
    all_centers = axial_sliding(center_list, entry, direction, bragg, step)

    # phase 3: greedily add centers where beam voxels are still uncovered
    centri_finali = residual_grabber(dose_up, all_centers, center_shape, dose_threshold)

    return np.array(centri_finali)
