"""
beam_tiling.py — Extract tiles along a proton beam from dose volumes.


Pipeline:
    1. Entry point  = boundary voxel with maximum dose
    2. Bragg peak   = voxel with maximum dose (argmax)
    3. Beam axis    = unit vector from entry to Bragg peak
    4. Beam mask    = cylindrical tube of radius `radius` around the axis
    5. Tile centres = equally spaced along the axis (step voxels apart)
"""
import numpy as np


def get_boundary_mask(dose, threshold_fraction=None):
    """
    Boolean mask of volume boundary voxels (the 6 faces).
    If threshold_fraction is given, only voxels with dose >= threshold are included.
    """
    shape    = dose.shape
    boundary = np.zeros(shape, dtype=bool)

    # mark True the 6 faces of the volume (edges on all 3 axes)
    boundary[0, :, :]  = True
    boundary[-1, :, :] = True
    boundary[:, 0, :]  = True
    boundary[:, -1, :] = True
    boundary[:, :, 0]  = True
    boundary[:, :, -1] = True

    if threshold_fraction is not None:
        # keep only boundary voxels with sufficiently high dose
        return boundary & (dose >= dose.max() * threshold_fraction)
    return boundary


def _find_entry_point(dose, threshold_fraction):
    """Find the boundary voxel with the maximum dose (beam entry point)."""
    shape      = dose.shape
    boundary   = get_boundary_mask(dose)
    candidates = boundary & (dose >= dose.max() * threshold_fraction)
    if not candidates.any():
        # no boundary voxel exceeds the threshold: use all boundary voxels
        candidates = boundary
    return np.array(np.unravel_index(np.argmax(dose * candidates), shape), dtype=float)


def _beam_direction(entry, bragg):
    """Compute the unit vector from the entry point to the Bragg peak."""
    vector = bragg - entry
    norm = np.linalg.norm(vector)
    if norm < 1e-6:
        # degenerate case: entry coincides with the Bragg peak (both on the boundary) — arbitrary direction
        return np.array([1.0, 0.0, 0.0])
    return vector / norm


def create_mask(dose, entry, direction, radius, threshold_fraction):
    """
    Beam mask: voxels inside the cylinder (distance <= radius from the axis)
    and with dose >= threshold_fraction * max(dose).
    """
    # build a grid with the coordinates of every voxel in the volume
    zz, yy, xx = np.mgrid[0:dose.shape[0], 0:dose.shape[1], 0:dose.shape[2]]
    coords      = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1).astype(float)

    # vector from entry point to each voxel
    voxel_vec = coords - entry

    # projection of each voxel onto the beam axis (distance along the beam)
    axis_proj = voxel_vec @ direction

    # perpendicular distance from each voxel to the beam axis (cylinder radius)
    perp_dist = np.linalg.norm(voxel_vec - np.outer(axis_proj, direction), axis=1)

    # voxels inside the cylinder: must be in front of the entry (proj >= 0) and within the radius
    in_cylinder   = (axis_proj >= 0) & (perp_dist <= radius)
    above_dose    = (dose.ravel() >= dose.max() * threshold_fraction)
    cylinder_mask = (in_cylinder & above_dose).reshape(dose.shape)

    # merge the cylindrical mask with high-dose boundary voxels
    return cylinder_mask | get_boundary_mask(dose, threshold_fraction)


def _extract_tile(volume, center, tile_shape):
    """Extract a 3D tile centered at center. Parts outside the volume are zero-padded."""
    center_z = int(round(center[0]))
    center_y = int(round(center[1]))
    center_x = int(round(center[2]))
    size_z   = tile_shape[0]
    size_y   = tile_shape[1]
    size_x   = tile_shape[2]

    # top-left corner of the tile in the volume (can be negative if it goes outside the border)
    origin = (center_z - size_z // 2, center_y - size_y // 2, center_x - size_x // 2)

    # empty array of tile size (automatic zero-padding for borders)
    tile = np.zeros((size_z, size_y, size_x), dtype=volume.dtype)

    vol_slices  = []
    tile_slices = []
    for axis, (origin_coord, axis_size) in enumerate(zip(origin, (size_z, size_y, size_x))):
        # clamp to volume borders
        vol_start = max(0, origin_coord)
        vol_end   = min(volume.shape[axis], origin_coord + axis_size)

        # offset inside the tile (>0 if the tile starts outside the volume on the left)
        tile_offset = vol_start - origin_coord

        vol_slices.append(slice(vol_start, vol_end))
        tile_slices.append(slice(tile_offset, tile_offset + vol_end - vol_start))

    # copy values from the volume into the corresponding portion of the tile
    tile[tuple(tile_slices)] = volume[tuple(vol_slices)]
    return tile, origin


def extract_dose_tiles(dose, tiles, tile_shape):
    """
    Extract raw dose tiles at the same positions as the already-computed beam tiles.
    Adds 'tile_dose' to each dict in the list in-place.

    Parameters
    ----------
    dose       : (Z, Y, X) ndarray — dose volume to sample
    tiles      : list[dict] — output of extract_beam_tiles (modified in-place)
    tile_shape : (tz, ty, tx)

    Returns
    -------
    tiles : same list with 'tile_dose' added to each dict
    """
    for t in tiles:
        tile_dose, _ = _extract_tile(dose, t['center'], tile_shape)
        t['tile_dose'] = tile_dose
    return tiles


def extract_beam_tiles(dose, tile_shape, step, threshold_fraction=0.1, radius=None):
    """
    Extract tiles along the beam axis.

    Parameters
    ----------
    dose               : (Z, Y, X) ndarray
    tile_shape         : (tz, ty, tx)
    step               : float — distance between adjacent tile centers (voxels)
    threshold_fraction : float — fraction of max dose for the beam mask
    radius             : float — beam mask cylinder radius (default: min(tile_shape)/4)

    Returns
    -------
    tiles     : list[dict]  {tile_id, center, origin, tile,
                              beam_coords_global, n_beam_voxels}
    mask      : bool ndarray — cylindrical beam mask
    entry     : (3,) entry point on the volume boundary
    direction : (3,) unit vector toward the Bragg peak
    """
    # find Bragg peak (absolute maximum) and entry point (maximum on the boundary)
    bragg     = np.array(np.unravel_index(np.argmax(dose), dose.shape), dtype=float)
    entry     = _find_entry_point(dose, threshold_fraction)
    direction = _beam_direction(entry, bragg)

    if radius is None:
        radius = min(tile_shape) / 4

    mask       = create_mask(dose, entry, direction, radius, threshold_fraction)
    mask_uint8 = mask.astype(np.uint8)
    beam_len   = float(np.linalg.norm(bragg - entry))  # total beam length in voxels

    tiles = []
    # place a tile center every `step` voxels along the beam axis
    for dist in np.arange(0.0, beam_len * 1.1, step):
        center = entry + dist * direction

        # skip if the center is outside the volume
        if not all(0 <= center[i] < dose.shape[i] for i in range(3)):
            continue

        tile_mask, origin = _extract_tile(mask_uint8, center, tile_shape)
        tile_mask         = tile_mask.astype(bool)

        # skip if the tile contains no beam mask voxels
        if not tile_mask.any():
            continue

        # local coordinates of beam voxels inside the tile
        beam_voxels_local = np.argwhere(tile_mask)

        tiles.append(dict(
            tile_id            = len(tiles),
            center             = tuple(int(round(float(v))) for v in center),
            origin             = origin,
            tile               = tile_mask,
            beam_coords_global = beam_voxels_local + np.array(origin),  # coordinates in the global volume
            n_beam_voxels      = len(beam_voxels_local),
        ))

    # fallback: the loop above can end up empty in degenerate cases (e.g. entry == bragg,
    # beam_len == 0) — force one tile centered on the Bragg peak so callers never see an empty list
    if not tiles:
        tile_mask, origin = _extract_tile(mask_uint8, bragg, tile_shape)
        tile_mask = tile_mask.astype(bool)
        beam_voxels_local = np.argwhere(tile_mask)
        tiles.append(dict(
            tile_id            = 0,
            center             = tuple(int(round(float(v))) for v in bragg),
            origin             = origin,
            tile               = tile_mask,
            beam_coords_global = beam_voxels_local + np.array(origin),
            n_beam_voxels      = len(beam_voxels_local),
        ))

    return tiles, mask, entry, direction
