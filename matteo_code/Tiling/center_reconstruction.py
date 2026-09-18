import numpy as np
from beam_tiling import _find_entry_point, _beam_direction, create_mask, _extract_tile
from reconstruction import trilinear_upsample, reconstruct_volume_from_tiles


def setup_beam(dose, threshold_fraction, radius, scale_factor=2):    # load radius and the data needed for the beam geometry
    """
    Load, upsample and compute beam geometry.

    Parameters
    ----------
    dose               : (Z, Y, X) ndarray — original dose volume
    threshold_fraction : float — dose threshold fraction (e.g. 0.1 = 10% of max)
    radius             : float — beam mask cylinder radius (voxels, in upsampled space)
    scale_factor       : int   — trilinear upscaling factor (default 2)

    Returns
    -------
    dose_up   : (Z*s, Y*s, X*s) ndarray — upsampled volume
    bragg     : (3,) array — Bragg peak coordinates in upsampled volume
    entry     : (3,) array — entry point on the volume boundary
    direction : (3,) array — unit vector entry → Bragg peak
    mask      : (Z*s, Y*s, X*s) bool ndarray — cylindrical beam mask
    """
    dose_up   = trilinear_upsample(dose, scale_factor=scale_factor)
    bragg     = np.array(np.unravel_index(np.argmax(dose_up), dose_up.shape), dtype=float)
    entry     = _find_entry_point(dose_up, threshold_fraction)
    direction = _beam_direction(entry, bragg)
    mask      = create_mask(dose_up, entry, direction, radius, threshold_fraction)

    return dose_up, bragg, entry, direction, mask


def check_center_fitting(tile_dose, tile_shape, center_shape, dose_threshold):  # check whether the tile center is large enough to contain the full beam
    """
    Check: is the tile center large enough to contain the entire beam?

    Checks whether all beam voxels in the tile fall within the center region.
    If there are beam voxels outside the center → expansion needed (returns False).
    If all beam voxels are inside the center → center is sufficient (returns True).

    Parameters
    ----------
    tile_dose      : (tz, ty, tx) ndarray — dose of the extracted tile
    tile_shape     : (tz, ty, tx) — full tile dimensions
    center_shape   : (cz, cy, cx) — central region dimensions
    dose_threshold : float — absolute threshold value (dose_up.max() * threshold_fraction)

    Returns
    -------
    bool — True if the center is large enough, False if lateral expansion is needed
    """
    offset_z = (tile_shape[0] - center_shape[0]) // 2
    offset_y = (tile_shape[1] - center_shape[1]) // 2
    offset_x = (tile_shape[2] - center_shape[2]) // 2

    # extract only the central region of the tile
    center_region = tile_dose[
        offset_z : offset_z + center_shape[0],
        offset_y : offset_y + center_shape[1],
        offset_x : offset_x + center_shape[2],
    ]

    # the beam fits in the center only if there is background at the edges (outer border of the center region)
    # check only the border voxels
    border_mask = np.zeros(center_region.shape, dtype=bool)
    border_mask[0, :, :]  = True
    border_mask[-1, :, :] = True
    border_mask[:, 0, :]  = True
    border_mask[:, -1, :] = True
    border_mask[:, :, 0]  = True
    border_mask[:, :, -1] = True

    return (center_region[border_mask] < dose_threshold).any()


def lateral_expansion(dose_up, beam_axis_point, direction, tile_shape, center_shape, dose_threshold):
    """
    Lateral expansion: finds the set of centers that covers the full beam cross-section.

    Starting from beam_axis_point, checks if a single center is enough. If not, iteratively
    adds adjacent centers in the direction of uncovered beam voxels until all are covered.
    Only voxels perpendicular to the beam direction are considered (cross-section).
    The returned centers are spaced by center_shape so they never overlap.

    Parameters
    ----------
    dose_up          : (Z, Y, X) ndarray — upsampled dose volume
    beam_axis_point  : (3,) array — point on the beam axis (global coordinates)
    direction        : (3,) array — unit vector entry → Bragg peak
    tile_shape       : (tz, ty, tx) — full tile dimensions
    center_shape     : (cz, cy, cx) — central region used for reconstruction
    dose_threshold   : float — absolute dose threshold (dose_up.max() * threshold_fraction)

    Returns
    -------
    centers : list of (3,) arrays — global coordinates of each center in the lateral pattern
    """
    tile_dose, origin = _extract_tile(dose_up, beam_axis_point, tile_shape)

    # for each voxel compute the distance from the tile center along the beam direction
    griglia_z, griglia_y, griglia_x = np.indices(tile_shape)      # voxel coordinate grid in the tile
    offset_z = griglia_z - tile_shape[0] / 2
    offset_y = griglia_y - tile_shape[1] / 2          # 0 at center, positive/negative forward or backward
    offset_x = griglia_x - tile_shape[2] / 2
    distanza_assiale = (offset_z * direction[0] +
                        offset_y * direction[1] +      # dot product for distance along the beam
                        offset_x * direction[2])
    cross_section = np.abs(distanza_assiale) <= 2.0   # tolerance for cross-section slice thickness

    tile_mid = np.array(tile_shape) / 2

    # all beam voxels in the plane perpendicular to the beam
    beam_cs_coords = np.argwhere((tile_dose >= dose_threshold) & cross_section)  # cross-section coordinates

    if len(beam_cs_coords) == 0:
        return [np.array(beam_axis_point, dtype=float)]  # return the single center if no other candidates exist

    # map each beam voxel to a grid cell of size center_shape;
    # each unique cell is a candidate center — no loop, no risk of infinite cycle
    celle_occupate = np.unique(
        np.round((beam_cs_coords - tile_mid) / np.array(center_shape, dtype=float)).astype(int),
        axis=0
    )    # list of occupied cells in the large tile of size center_shape; each cell is a candidate center; more than one means the single center is not enough

    beam_axis_point = np.array(beam_axis_point, dtype=float)
    center_shape    = np.array(center_shape,    dtype=float)
    centers = beam_axis_point + celle_occupate * center_shape # global coordinates of each center: start from tile center and shift by occupied cell * center size
    return list(centers)


def axial_sliding(center_list, entry, direction, bragg, step):   # if step <= center_shape[dominant axis], centers will be adjacent; if equal, they will be exactly touching
    """
    Axial sliding: translates the lateral pattern along the beam axis from entry to Bragg peak.

    Takes the set of centers found by lateral_expansion and slides them along the beam
    direction at regular intervals of step, until the Bragg peak is reached.

    Parameters
    ----------
    center_list : list of (3,) arrays — lateral pattern centers (global coordinates)
    entry       : (3,) array — entry point on the volume boundary
    direction   : (3,) array — unit vector entry → Bragg peak
    bragg       : (3,) array — Bragg peak coordinates
    step        : float — axial spacing between successive positions (voxels)

    Returns
    -------
    all_centers : list of (3,) arrays — all centers covering the full beam volume
    """
    beam_length = np.linalg.norm(bragg - entry)
    all_centers = []
    dist = 0.0

    while dist <= beam_length:              # keep sliding until the Bragg peak is reached
        axis_offset = direction * dist      # displacement vector along the beam axis
        for c in center_list:
            all_centers.append(c + axis_offset)
        dist += step

    return [np.round(c).astype(float) for c in all_centers]


def reconstruct_from_centers(dose_up, all_centers, center_shape):
    tiles_estratti    = []                                        # list of tiles to reconstruct
    dimensioni_volume = np.array(dose_up.shape)                   # total volume dimensions (Z, Y, X)
    dimensioni_centro = np.array(center_shape)                    # central region dimensions

    for centro in all_centers:
        # compute where the tile starts and ends in the volume
        inizio_tile = np.round(np.array(centro) - dimensioni_centro / 2).astype(int)  # bottom-left corner
        fine_tile   = inizio_tile + dimensioni_centro                                  # top-right corner

        # skip centers whose tile lies completely outside the volume (avoids crash in _extract_tile)
        if np.any(fine_tile <= 0) or np.any(inizio_tile >= dimensioni_volume):
            continue

        dose_centro, origine = _extract_tile(dose_up, centro, center_shape)   # extract the dose of the central region
        tiles_estratti.append({'tile_dose': dose_centro, 'origin': origine})  # add to list

    return reconstruct_volume_from_tiles(tiles_estratti, center_shape, dose_up.shape)  # reconstruct the volume from pieces


def residual_grabber(dose_up, all_centers, center_shape, dose_threshold):
    """
    Greedy residual coverage: finds centers left uncovered after axial_sliding.

    Starts from the original volume, zeroes out regions already covered by all_centers,
    then adds new centers one at a time until the residual maximum drops
    below the threshold.

    Returns the complete list of centers (original + new).
    """
    dimensioni_volume = np.array(dose_up.shape)   # total volume dimensions (Z, Y, X)
    dimensioni_centro = np.array(center_shape)    # central region dimensions

    # copy of the volume where already-covered zones are progressively zeroed
    residuo = dose_up.copy()

    # zero out regions already covered by all_centers
    for centro in all_centers:
        inizio      = np.round(np.array(centro) - dimensioni_centro / 2).astype(int)
        fine        = inizio + dimensioni_centro              # define the tile to zero around the center
        inizio_clip = np.maximum(inizio, 0)                   # clamp to lower border
        fine_clip   = np.minimum(fine, dimensioni_volume)     # clamp to upper border
        residuo[inizio_clip[0]:fine_clip[0],
                inizio_clip[1]:fine_clip[1],
                inizio_clip[2]:fine_clip[2]] = 0    # zero out the center tile

    # greedy loop: keep adding centers as long as significant uncovered dose remains
    nuovi_centri = []
    while residuo.max() >= dose_threshold:
        picco = np.array(np.unravel_index(np.argmax(residuo), residuo.shape), dtype=float)  # voxel with highest remaining dose
        nuovi_centri.append(picco)  # add it as a new center

        # zero out the region around the peak so it is not picked again next iteration
        inizio      = np.round(picco - dimensioni_centro / 2).astype(int)
        fine        = inizio + dimensioni_centro
        inizio_clip = np.maximum(inizio, 0)
        fine_clip   = np.minimum(fine, dimensioni_volume)
        residuo[inizio_clip[0]:fine_clip[0],
                inizio_clip[1]:fine_clip[1],
                inizio_clip[2]:fine_clip[2]] = 0  # zero out: this zone has just been claimed

    return list(all_centers) + nuovi_centri  # return all centers: original + new
