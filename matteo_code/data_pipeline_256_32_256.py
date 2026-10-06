import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from Tiling.beam_tiling import extract_beam_tiles, _extract_tile

class data_pipeline(Dataset):
    """
    PyTorch Dataset for 3D CT and dose volumes.
    Binary mask is downsampled and returned separately.
    """
    def __init__(self, path, index_list, mode, tile_shape, threshold=0.1, refine=False):
        """
        path: folder containing patient subfolders
        index_list: list of patient IDs
        mask_downsample_factor: int, factor to downsample mask
        use_mask: whether to return mask
        mode: 'none' (whole volume, default) | 'tile' (every candidate tile for the
              patient, stacked)
        tile_shape: (tz, ty, tx) tile size, used only when mode='tile' — keep divisible by 8
        """
        self.path = path
        self.index_list = index_list
        self.threshold = threshold
        self.refine = refine
        self.mode = mode
        self.tile_shape = tile_shape

    def __len__(self):
        return len(self.index_list)*4

    def __getitem__(self, idx):

        base_idx = idx//4
        rot_type = idx%4 # 0:no rot, 1:90deg, 2:180deg, 3:270deg
        
        #ID = self.index_list[base_idx]
        #folder = os.path.join(self.path, str(ID))

        # --- Load raw data ---

        # /media/proton-lab/EXTERNAL_USB/matteo_thesis/data/<split>/<patient_id>/CT.npy | dose5K.npy | dose1M.npy
        patient_folder = self.path + str(self.index_list[base_idx]) + '/'
        CT = np.load(patient_folder + 'CT.npy').astype(np.float32)
        Dose_5K = np.load(patient_folder + 'dose5K.npy').astype(np.float32)
        Dose_1M = np.load(patient_folder + 'dose1M.npy').astype(np.float32)


        # --- Normalize CT ---
        CT_norm = (CT + 1000.0) / 3000.0  # scale ~0-1
        Dose_5K_norm = (Dose_5K - np.min(Dose_5K)) / (np.max(Dose_5K) - np.min(Dose_5K))
        Dose_1M_norm = (Dose_1M - np.min(Dose_1M)) / (np.max(Dose_1M) - np.min(Dose_1M))
     
        if self.refine:
            threshold_value = self.threshold * np.max(Dose_5K_norm)
            Dose_5K_norm = np.where(Dose_5K_norm < threshold_value, 0.0, Dose_5K_norm)

        if rot_type !=0:

            # Rotate around axis 0 (the elongated beam-depth axis, e.g. 256) using the
            # plane formed by axes 1 and 2 (equal-sized, e.g. 32x32) so the shape stays
            # constant across all 4 rotations -- axes (0,2) would swap unequal axis sizes.
            CT_norm_rotated = np.rot90(CT_norm, k=rot_type, axes=(1,2)).copy()
            Dose_5K_norm_rotated = np.rot90(Dose_5K_norm, k=rot_type, axes=(1,2)).copy()
            Dose_1M_norm_rotated = np.rot90(Dose_1M_norm, k=rot_type, axes=(1,2)).copy()

        else:
            CT_norm_rotated = CT_norm
            Dose_5K_norm_rotated = Dose_5K_norm
            Dose_1M_norm_rotated = Dose_1M_norm

        # --- Normalize Dose_5K ---
        #Dose_5K_norm = (Dose_5K - Dose_5K.min()) / (Dose_5K.max() - Dose_5K.min() + 1e-8)

        # --- mode='tile': every candidate tile for this patient, stacked on a leading
        #     dimension (acts as the batch dimension for this patient's forward pass) ---
        if self.mode == 'tile':
            # step = tile_shape[0] (the beam-depth extent): tile centers spaced so
            # adjacent tiles touch edge-to-edge along the beam axis with no overlap
            # (the only axis extract_beam_tiles places multiple tiles along -- laterally
            # every tile is already centered on the beam axis, nothing to tile there).
            step = self.tile_shape[0]
            tiles, _, _, _ = extract_beam_tiles(
                Dose_5K_norm_rotated, self.tile_shape, step, threshold_fraction=self.threshold
            )

            ct_tiles, dose5k_tiles, dose1m_tiles = [], [], []
            for t in tiles:
                center = t['center']
                ct_t, _ = _extract_tile(CT_norm_rotated,      center, self.tile_shape)
                d5_t, _ = _extract_tile(Dose_5K_norm_rotated, center, self.tile_shape)
                d1_t, _ = _extract_tile(Dose_1M_norm_rotated, center, self.tile_shape)
                ct_tiles.append(ct_t)
                dose5k_tiles.append(d5_t)
                dose1m_tiles.append(d1_t)

            CT_stack    = np.stack(ct_tiles, axis=0)      # (n_tiles, Tz, Ty, Tx)
            Dose5K_stack = np.stack(dose5k_tiles, axis=0)  # (n_tiles, Tz, Ty, Tx)
            Dose1M_stack = np.stack(dose1m_tiles, axis=0)  # (n_tiles, Tz, Ty, Tx)

            X = torch.from_numpy(np.stack((CT_stack, Dose5K_stack), axis=1))  # (n_tiles, 2, Tz, Ty, Tx)
            Y = torch.from_numpy(np.expand_dims(Dose1M_stack, axis=1))        # (n_tiles, 1, Tz, Ty, Tx)
            return X, Y

        # --- mode='none': it doesn't change the pipeline ---

        # --- Binary mask ---

        # --- Stack input channels ---
        X= torch.from_numpy(np.stack((CT_norm_rotated, Dose_5K_norm_rotated), axis=0))
        Y = torch.from_numpy(np.expand_dims(Dose_1M_norm_rotated, axis=0))


        return X, Y