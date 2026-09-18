import numpy as np
from beam_tiling import _extract_tile
from center_reconstruction import setup_beam, axial_sliding, residual_grabber


def lateral_expansion_stride(dose_up, beam_axis_point, direction, tile_shape, lateral_step, dose_threshold):
    """
    Like lateral_expansion but uses lateral_step as the lateral grid spacing,
    allowing overlap between lateral tiles if lateral_step < center_shape.

    Parameters
    ----------
    dose_up         : (Z, Y, X) ndarray — upsampled dose volume
    beam_axis_point : (3,) array — point on the beam axis
    direction       : (3,) array — unit vector entry → bragg
    tile_shape      : (tz, ty, tx) — large tile dimensions
    lateral_step    : (sz, sy, sx) — lateral step size
    dose_threshold  : float — dose threshold

    Returns
    -------
    centers : list of (3,) arrays — centers of the lateral pattern
    """
    tile_dose, _ = _extract_tile(dose_up, beam_axis_point, tile_shape)

    # cross-section perpendicular to the beam
    griglia_z, griglia_y, griglia_x = np.indices(tile_shape)
    offset_z = griglia_z - tile_shape[0] / 2
    offset_y = griglia_y - tile_shape[1] / 2
    offset_x = griglia_x - tile_shape[2] / 2
    distanza_assiale = (offset_z * direction[0] +
                        offset_y * direction[1] +
                        offset_x * direction[2])
    cross_section = np.abs(distanza_assiale) <= 2.0

    tile_mid = np.array(tile_shape) / 2
    beam_cs_coords = np.argwhere((tile_dose >= dose_threshold) & cross_section)

    if len(beam_cs_coords) == 0:
        return [np.array(beam_axis_point, dtype=float)]

    # grid-snap with lateral_step instead of center_shape → tiles can overlap
    celle_occupate = np.unique(
        np.round((beam_cs_coords - tile_mid) / np.array(lateral_step, dtype=float)).astype(int),
        axis=0
    )

    beam_axis_point = np.array(beam_axis_point, dtype=float)
    lateral_step    = np.array(lateral_step,    dtype=float)
    centers = beam_axis_point + celle_occupate * lateral_step
    return list(centers)


def run_pipeline_stride(dose, tile_shape, center_shape, axial_step, lateral_step,
                        threshold_fraction, radius):
    """
    Complete pipeline with configurable stride in all 3 dimensions.

    Parameters
    ----------
    dose               : (Z, Y, X) ndarray — original dose volume
    tile_shape         : (tz, ty, tx) — large tile size for lateral expansion
    center_shape       : (cz, cy, cx) — central region size for reconstruction
    axial_step         : float — step along the beam axis (if < center_shape[1] → axial overlap)
    lateral_step       : (sz, sy, sx) — lateral step (if < center_shape → lateral overlap)
    threshold_fraction : float — dose threshold as a fraction of the maximum
    radius             : float — beam mask cylinder radius

    Returns
    -------
    centri_finali : (N, 3) ndarray — global coordinates of all centers
    """
    dose_up, bragg, entry, direction, _ = setup_beam(dose, threshold_fraction, radius)
    dose_threshold = dose_up.max() * threshold_fraction

    center_list = lateral_expansion_stride(dose_up, entry, direction, tile_shape,
                                           lateral_step, dose_threshold)
    all_centers = axial_sliding(center_list, entry, direction, bragg, axial_step)
    centri_finali = residual_grabber(dose_up, all_centers, center_shape, dose_threshold)

    return np.array(centri_finali)
