"""
Test a tile-trained checkpoint by reconstructing the FULL prediction from ALL of
a patient's candidate tiles (not a single random one), then computing WMSE/GPR
only on the voxels that were actually covered by at least one reconstructed tile
(not on the whole 256x32x32 volume, which would include untouched background).
"""
import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network_2_1 import unet
from Tiling.beam_tiling import extract_beam_tiles, _extract_tile
from Tiling.reconstruction import reconstruct_volume_from_tiles
from losses_opt_1mm import GammaIndexLoss, WeightedMSE

np.random.seed(42)
torch.manual_seed(42)

# ===================== CONFIG: cambia questi prima di ogni test =====================
MODEL_PATH = "<incolla il path del checkpoint>"
TILE_SHAPE = (128, 16, 16)   # deve combaciare con la TILE_SHAPE usata in training per QUESTO checkpoint
config_tag = "tile_gamma_reconstructed"    # solo per nominare gli output
TILE_SUBBATCH = 8            # tile processate insieme per forward pass (solo per controllare la VRAM)
# ======================================================================================

input_shape = (256, 32, 256)

DOSE_PERCENT_THRESHOLD = 2.0
DTA_MM_THRESHOLD = 2.0
VOXEL_SIZE_MM = 2.0
DOSE_CUTOFF = 0.2
WMSE_ALPHA = 1.0

test_path = '/media/proton-lab/EXTERNAL_USB/matteo_thesis/data/Test/'
output_root = f"/media/proton-lab/EXTERNAL_USB/matteo_thesis/models_and_outputs/test_results/{config_tag}"
image_folder = os.path.join(output_root, "images")
results_csv = os.path.join(output_root, "test_results.csv")
os.makedirs(image_folder, exist_ok=True)

device = "cuda"

test_examples = len([d for d in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, d))])
index_list_test = [str(i) for i in range(test_examples)]
print(f"Found {test_examples} test patients")

# ===================== MODEL =====================
model = unet(input_shape=input_shape)
model = model.to(device)

dummy_channels = 2  # CT + Dose5K
dummy = torch.randn(1, dummy_channels, *input_shape, device=device)
with torch.no_grad():
    _ = model(dummy)
del dummy

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()
print(f"Loaded checkpoint: {MODEL_PATH}")

# ===================== METRICS (gamma_loss used directly so we can intersect
# its dose-threshold mask with the tile-coverage mask before computing GPR) =====================
wmse_loss = WeightedMSE(alpha=WMSE_ALPHA).to(device)
gamma_loss = GammaIndexLoss(
    dose_percent=DOSE_PERCENT_THRESHOLD,
    dta_mm=DTA_MM_THRESHOLD,
    voxel_size_mm=(VOXEL_SIZE_MM, VOXEL_SIZE_MM, VOXEL_SIZE_MM),
    dose_cutoff=DOSE_CUTOFF,
    beta_init=5.0,
    max_gamma=10.0,
).to(device)


def load_patient(pid):
    folder = os.path.join(test_path, pid) + '/'
    CT = np.load(folder + 'CT.npy').astype(np.float32)
    Dose_5K = np.load(folder + 'dose5K.npy').astype(np.float32)
    Dose_1M = np.load(folder + 'dose1M.npy').astype(np.float32)
    CT_norm = (CT + 1000.0) / 3000.0
    Dose_5K_norm = (Dose_5K - Dose_5K.min()) / (Dose_5K.max() - Dose_5K.min())
    Dose_1M_norm = (Dose_1M - Dose_1M.min()) / (Dose_1M.max() - Dose_1M.min())
    return CT_norm, Dose_5K_norm, Dose_1M_norm


# ===================== TEST LOOP =====================
rows = []
with torch.no_grad(), open(results_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['patient_id', 'n_tiles', 'coverage_voxels', 'wmse', 'gpr_percent'])

    for i, pid in enumerate(index_list_test):
        CT_norm, Dose_5K_norm, Dose_1M_norm = load_patient(pid)
        full_shape = Dose_5K_norm.shape

        step = min(TILE_SHAPE) / 2
        tiles, _, _, _ = extract_beam_tiles(Dose_5K_norm, TILE_SHAPE, step, threshold_fraction=0.1)

        ct_tiles, d5_tiles = [], []
        for t in tiles:
            ct_t, _ = _extract_tile(CT_norm, t['center'], TILE_SHAPE)
            d5_t, _ = _extract_tile(Dose_5K_norm, t['center'], TILE_SHAPE)
            ct_tiles.append(ct_t)
            d5_tiles.append(d5_t)
        X = torch.from_numpy(np.stack((np.stack(ct_tiles), np.stack(d5_tiles)), axis=1)).to(device)  # (n_tiles,2,Tz,Ty,Tx)
        n_tiles = X.size(0)

        pred_tiles = []
        with autocast(dtype=torch.bfloat16):
            for start in range(0, n_tiles, TILE_SUBBATCH):
                sub_X = X[start:start + TILE_SUBBATCH]
                output = model(x=sub_X)
                output_norm = output / (output.amax(dim=(2, 3, 4), keepdim=True) + 1e-10)
                pred_np = output_norm[:, 0].float().cpu().numpy()  # (sub_n, Tz,Ty,Tx)
                for j in range(pred_np.shape[0]):
                    pred_tiles.append({'origin': tiles[start + j]['origin'], 'tile_dose': pred_np[j]})

        pred_volume, coverage = reconstruct_volume_from_tiles(pred_tiles, TILE_SHAPE, full_shape, return_coverage=True)

        pred_t = torch.from_numpy(pred_volume).unsqueeze(0).unsqueeze(0).to(device)    # (1,1,Z,Y,X)
        target_t = torch.from_numpy(Dose_1M_norm).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,Z,Y,X)
        coverage_t = torch.from_numpy(coverage).unsqueeze(0).unsqueeze(0).to(device)    # (1,1,Z,Y,X)

        # --- GPR restricted to tile coverage ---
        gamma, dose_valid_mask = gamma_loss.compute_gamma(pred_t, target_t)
        # compute_gamma resamples to ~1mm internally (voxel_size_mm), so gamma/dose_valid_mask
        # live on a different grid than coverage_t -- resample coverage (nearest, it's boolean) to match
        if gamma.shape[2:] != coverage_t.shape[2:]:
            coverage_resampled = F.interpolate(coverage_t.float(), size=gamma.shape[2:], mode='nearest') > 0.5
        else:
            coverage_resampled = coverage_t
        final_mask = dose_valid_mask & coverage_resampled
        n_valid = final_mask.float().sum()
        if n_valid == 0:
            gpr = 100.0
        else:
            passing = (gamma < 1.0) & final_mask
            gpr = (passing.float().sum() / n_valid).item() * 100

        # --- WMSE restricted to tile coverage (same weighting formula as WeightedMSE, masked) ---
        weights = torch.exp(WMSE_ALPHA * torch.clamp(target_t, -10.0, 10.0))
        sq_error = (pred_t - target_t) ** 2
        mask_f = coverage_t.float()
        wmse = ((weights * sq_error * mask_f).sum() / ((weights * mask_f).sum() + 1e-8)).item()

        coverage_voxels = int(coverage.sum())
        writer.writerow([pid, n_tiles, coverage_voxels, wmse, gpr])
        rows.append((wmse, gpr))
        print(f"[{i+1}/{test_examples}] patient {pid}: n_tiles={n_tiles}, coverage={coverage_voxels} vox, WMSE={wmse:.6f}, GPR={gpr:.2f}%")

        # ---- Save comparison image (central slice along the short axis) ----
        mid = pred_volume.shape[1] // 2
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(Dose_1M_norm[:, mid, :], cmap='jet', vmin=0, vmax=1)
        axes[0].set_title("Target (full volume)")
        axes[1].imshow(pred_volume[:, mid, :], cmap='jet', vmin=0, vmax=1)
        axes[1].set_title("Predicted (reconstructed from tiles)")
        diff = np.where(coverage, pred_volume - Dose_1M_norm, np.nan)
        axes[2].imshow(diff[:, mid, :], cmap='bwr', vmin=-0.3, vmax=0.3)
        axes[2].set_title("Difference (coverage only)")
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"Patient {pid} | GPR {gpr:.2f}% | WMSE {wmse:.4f} | {n_tiles} tile")
        fig.tight_layout()
        fig.savefig(os.path.join(image_folder, f"patient_{pid}.png"), dpi=120)
        plt.close(fig)

    arr = np.array(rows)
    means = arr.mean(axis=0)
    stds = arr.std(axis=0)
    writer.writerow([])
    writer.writerow(['MEAN', '', '', means[0], means[1]])
    writer.writerow(['STD', '', '', stds[0], stds[1]])

print("\n" + "=" * 50)
print(f"Test complete on {len(rows)} patients")
print(f"Mean WMSE: {means[0]:.6f} | Mean GPR: {means[1]:.2f}% (+/-{stds[1]:.2f})")
print(f"Results CSV: {results_csv}")
print(f"Images: {image_folder}")
