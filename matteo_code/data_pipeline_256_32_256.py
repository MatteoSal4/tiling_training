import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from Tiling.beam_tiling import extract_beam_tiles, extract_dose_tiles, _extract_tile
from Tiling.reconstruction import reconstruct_volume_from_tiles

MODE = 'none'  # cambio per test  (none tile o rec)
TILE_SHAPE = (64, 16, 64)  # cambio per test (divisibile per 8 sennò errore)

class data_pipeline(Dataset):
    """
    PyTorch Dataset for 3D CT and dose volumes.
    Binary mask is downsampled and returned separately.
    """
    def __init__(self, path, index_list, threshold=0.1, refine=False,
                 mode=MODE, tile_shape=TILE_SHAPE, radius=None):
        """
        path: folder containing patient subfolders
        index_list: list of patient IDs
        mask_downsample_factor: int, factor to downsample mask
        use_mask: whether to return mask
        mode: 'none' (whole volume, default) | 'tile' (single small tile) | 'rec' (reconstructed volume from tiles)
        tile_shape: (tz, ty, tx) tile size, used only when mode='tile'/'rec' — keep divisible by 8
        radius: beam mask cylinder radius; None -> min(tile_shape)/4
        """
        self.path = path
        self.index_list = index_list
        self.threshold = threshold
        self.refine = refine
        self.mode = mode
        self.tile_shape = tile_shape
        self.radius = radius
       
    def __len__(self):
        return len(self.index_list)*4

    def __getitem__(self, idx):

        base_idx = idx//4
        rot_type = idx%4 # 0:no rot, 1:90deg, 2:180deg, 3:270deg
        
        #ID = self.index_list[base_idx]
        #folder = os.path.join(self.path, str(ID))

        # --- Load raw data ---

        # /media/proton-lab/migrameter_/data/<split>/<patient_id>/CT.npy | dose5K.npy | dose1M.npy
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

            CT_norm_rotated = np.rot90(CT_norm, k=rot_type, axes=(0,2)).copy()
            Dose_5K_norm_rotated = np.rot90(Dose_5K_norm, k=rot_type, axes=(0,2)).copy()
            Dose_1M_norm_rotated = np.rot90(Dose_1M_norm, k=rot_type, axes=(0,2)).copy()

        else:
            CT_norm_rotated = CT_norm
            Dose_5K_norm_rotated = Dose_5K_norm
            Dose_1M_norm_rotated = Dose_1M_norm

        # --- Normalize Dose_5K ---
        #Dose_5K_norm = (Dose_5K - Dose_5K.min()) / (Dose_5K.max() - Dose_5K.min() + 1e-8)

        # --- mode='tile': it uses the tile instead of the whole volume ---
        if self.mode == 'tile':
            step = min(self.tile_shape) / 2
            tiles, _, _, _ = extract_beam_tiles(
                Dose_5K_norm_rotated, self.tile_shape, step, threshold_fraction=self.threshold
            )
            chosen = tiles[np.random.randint(len(tiles))]
            center = chosen['center']

            CT_norm_rotated, _      = _extract_tile(CT_norm_rotated,      center, self.tile_shape)
            Dose_5K_norm_rotated, _ = _extract_tile(Dose_5K_norm_rotated, center, self.tile_shape)
            Dose_1M_norm_rotated, _ = _extract_tile(Dose_1M_norm_rotated, center, self.tile_shape)

        # --- mode='rec': it reconstructs the volume from the tiles ---
        elif self.mode == 'rec':
            step = min(self.tile_shape) / 2
            tiles, _, _, _ = extract_beam_tiles(
                Dose_5K_norm_rotated, self.tile_shape, step, threshold_fraction=self.threshold, radius=self.radius
            )

            extract_dose_tiles(Dose_5K_norm_rotated, tiles, self.tile_shape)
            Dose_5K_norm_rotated = reconstruct_volume_from_tiles(tiles, self.tile_shape, Dose_5K_norm_rotated.shape)

            extract_dose_tiles(CT_norm_rotated, tiles, self.tile_shape)
            CT_norm_rotated = reconstruct_volume_from_tiles(tiles, self.tile_shape, CT_norm_rotated.shape)

            # Dose_1M_norm_rotated stays untouched

        # --- mode='none': it doesn't change the pipeline ---

        # --- Binary mask ---

        # --- Stack input channels ---
        X= torch.from_numpy(np.stack((CT_norm_rotated, Dose_5K_norm_rotated), axis=0))
        Y = torch.from_numpy(np.expand_dims(Dose_1M_norm_rotated, axis=0))


        return X, Y