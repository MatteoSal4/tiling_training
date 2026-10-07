"""
Test a trained checkpoint on the Test set: per-patient WMSE / gamma pass rate,
a summary CSV, and target-vs-predicted-vs-difference images.

Used for ALL checkpoints, whatever mode they were trained with -- only the
CONFIG flags below need to change:
- USE_TILING='none': single forward pass per patient on the whole volume
  (data_pipeline handles it directly).
- USE_TILING='tile': the checkpoint was trained on ALL of a patient's tiles
  (non-overlapping). Here we extract every candidate tile for each test
  patient, run them through the model, reconstruct the full predicted volume
  from the tile predictions (zero where no tile is present), and compute
  WMSE/GPR on the whole volume against the full 1M dose.
"""
import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network_2_1 import unet
from data_pipeline_256_32_256 import data_pipeline
from Tiling.beam_tiling import extract_beam_tiles, _extract_tile
from Tiling.reconstruction import reconstruct_volume_from_tiles
from losses_opt_1mm import GammaIndexLoss, WeightedMSE, CombinedWMSEGammaLoss

np.random.seed(42)
torch.manual_seed(42)

# ===================== CONFIG: cambia questi prima di ogni test =====================
MODEL_PATH = "/media/proton-lab/EXTERNAL_USB/matteo_thesis/models_and_outputs/2026-10-06_18-58-36_tile_gamma_combined_wmse_gamma_train_2_1_out_256-32-256_zero_outside_more_data_2mm_1%.pth"
USE_TILING = 'tile'          # None (volume intero) | 'tile' (tutte le tile, ricostruzione)
TILE_SHAPE = (128, 16, 16)   # deve combaciare con la TILE_SHAPE usata in fase di training per QUESTO checkpoint
config_tag = "tile_gamma_fullvol"    # solo per nominare gli output
TILE_SUBBATCH = 8            # mode='tile' only: tile processate insieme per forward pass (solo per controllare la VRAM)
# ======================================================================================

DATA_MODE = USE_TILING if USE_TILING else 'none'
input_shape = (256, 32, 256)

DOSE_PERCENT_THRESHOLD = 2.0
DTA_MM_THRESHOLD = 2.0
VOXEL_SIZE_MM = 2.0

test_path = '/media/proton-lab/EXTERNAL_USB/matteo_thesis/data/Test/'
output_root = f"/media/proton-lab/EXTERNAL_USB/matteo_thesis/models_and_outputs/test_results/{config_tag}"
image_folder = os.path.join(output_root, "images")
results_csv = os.path.join(output_root, "test_results.csv")
os.makedirs(image_folder, exist_ok=True)

device = "cuda"

test_examples = len([d for d in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, d))])
index_list_test = np.array([str(i) for i in range(test_examples)])
print(f"Found {test_examples} test patients")

# ===================== MODEL =====================
model = unet(input_shape=input_shape)
model = model.to(device)

input_channels = 2  # CT + Dose5K, same for every mode
dummy = torch.randn(1, input_channels, *input_shape, device=device)
with torch.no_grad():
    _ = model(dummy)
del dummy

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()
print(f"Loaded checkpoint: {MODEL_PATH}")

# ===================== LOSS / METRICS =====================
wmse_loss = WeightedMSE(alpha=1.0).to(device)
gamma_loss = GammaIndexLoss(
    dose_percent=DOSE_PERCENT_THRESHOLD,
    dta_mm=DTA_MM_THRESHOLD,
    voxel_size_mm=(VOXEL_SIZE_MM, VOXEL_SIZE_MM, VOXEL_SIZE_MM),
    dose_cutoff=0.2,
    beta_init=5.0,   # beta finale raggiunto a fine training; non influisce sulla pass rate "hard"
    max_gamma=10.0,
).to(device)
criterion = CombinedWMSEGammaLoss(
    wmse_loss=wmse_loss,
    gamma_loss=gamma_loss,
    wmse_weight=1.0,
    gamma_weight=1.0,
    outside_weight=0.0,
    outside_threshold=0.0,
    momentum=0.99,
).to(device)


# ===================== TEST LOOP: mode='tile' (ricostruzione da tutte le tile) =====================
def run_tile_mode():
    def load_patient(pid):
        folder = os.path.join(test_path, pid) + '/'
        CT = np.load(folder + 'CT.npy').astype(np.float32)
        Dose_5K = np.load(folder + 'dose5K.npy').astype(np.float32)
        Dose_1M = np.load(folder + 'dose1M.npy').astype(np.float32)
        CT_norm = (CT + 1000.0) / 3000.0
        Dose_5K_norm = (Dose_5K - Dose_5K.min()) / (Dose_5K.max() - Dose_5K.min())
        Dose_1M_norm = (Dose_1M - Dose_1M.min()) / (Dose_1M.max() - Dose_1M.min())
        return CT_norm, Dose_5K_norm, Dose_1M_norm

    rows = []
    with torch.no_grad(), open(results_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['patient_id', 'n_tiles', 'coverage_voxels', 'wmse', 'gpr_percent'])

        for i, pid in enumerate(index_list_test):
            CT_norm, Dose_5K_norm, Dose_1M_norm = load_patient(pid)
            full_shape = Dose_5K_norm.shape

            step = TILE_SHAPE[0]  # stesso step non sovrapposto usato in training per 'tile'
            tiles, _, _, _ = extract_beam_tiles(Dose_5K_norm, TILE_SHAPE, step, threshold_fraction=0.1)

            ct_tiles, d5_tiles = [], []
            for t in tiles:
                ct_t, _ = _extract_tile(CT_norm, t['center'], TILE_SHAPE)
                d5_t, _ = _extract_tile(Dose_5K_norm, t['center'], TILE_SHAPE)
                ct_tiles.append(ct_t)
                d5_tiles.append(d5_t)
            X = torch.from_numpy(np.stack((np.stack(ct_tiles), np.stack(d5_tiles)), axis=1)).to(device)
            n_tiles = X.size(0)

            pred_tiles = []
            with autocast(dtype=torch.bfloat16):
                for start in range(0, n_tiles, TILE_SUBBATCH):
                    sub_X = X[start:start + TILE_SUBBATCH]
                    output = model(x=sub_X)
                    output_norm = output / (output.amax(dim=(2, 3, 4), keepdim=True) + 1e-10)
                    pred_np = output_norm[:, 0].float().cpu().numpy()
                    for j in range(pred_np.shape[0]):
                        pred_tiles.append({'origin': tiles[start + j]['origin'], 'tile_dose': pred_np[j]})

            pred_volume, coverage = reconstruct_volume_from_tiles(pred_tiles, TILE_SHAPE, full_shape, return_coverage=True)

            pred_t = torch.from_numpy(pred_volume).unsqueeze(0).unsqueeze(0).to(device)
            target_t = torch.from_numpy(Dose_1M_norm).unsqueeze(0).unsqueeze(0).to(device)
            coverage_t = torch.from_numpy(coverage).unsqueeze(0).unsqueeze(0).to(device)

            # GPR on the whole volume: voxels not covered by any tile are zero in the
            # prediction, so those above the dose cutoff in the target count as failures.
            gamma, dose_valid_mask = gamma_loss.compute_gamma(pred_t, target_t)
            final_mask = dose_valid_mask
            n_valid = final_mask.float().sum()
            if n_valid == 0:
                gpr = 100.0
            else:
                passing = (gamma < 1.0) & final_mask
                gpr = (passing.float().sum() / n_valid).item() * 100

            # WMSE on the whole volume (same weighting formula as WeightedMSE)
            weights = torch.exp(torch.clamp(target_t, -10.0, 10.0))  # alpha=1.0
            sq_error = (pred_t - target_t) ** 2
            wmse = ((weights * sq_error).sum() / (weights.sum() + 1e-8)).item()

            coverage_voxels = int(coverage.sum())
            writer.writerow([pid, n_tiles, coverage_voxels, wmse, gpr])
            rows.append((wmse, gpr))
            print(f"[{i+1}/{test_examples}] patient {pid}: n_tiles={n_tiles}, coverage={coverage_voxels} vox, WMSE={wmse:.6f}, GPR={gpr:.2f}%")

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


# ===================== TEST LOOP: mode='none' (volume singolo per paziente) =====================
def run_volume_mode():
    class NoAugmentTestSet(Dataset):
        """Wraps data_pipeline to drop the x4 rotation augmentation: one deterministic
        (no-rotation) sample per patient, so test results don't vary between runs."""
        def __init__(self, base_dataset, n_patients):
            self.base = base_dataset
            self.n_patients = n_patients

        def __len__(self):
            return self.n_patients

        def __getitem__(self, idx):
            return self.base[idx * 4]  # rot_type 0 -> no rotation

    base_test_dataset = data_pipeline(
        path=test_path,
        index_list=index_list_test,
        threshold=0.1,
        mode=DATA_MODE,
        tile_shape=TILE_SHAPE,
    )
    test_dataset = NoAugmentTestSet(base_test_dataset, test_examples)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    rows = []
    with torch.no_grad(), open(results_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['patient_id', 'wmse', 'gamma', 'gpr_percent'])

        for i, (X, Y) in enumerate(test_dataloader):
            input = X.to(device, non_blocking=True)
            target = Y.to(device, non_blocking=True)

            with autocast(dtype=torch.bfloat16):
                output = model(x=input)
                output_norm = output / (output.amax(dim=(2, 3, 4), keepdim=True) + 1e-10)
                _, loss_dict = criterion(output_norm, target)
                gpr = criterion.compute_pass_rate(output_norm, target)

            writer.writerow([index_list_test[i], loss_dict['wmse'], loss_dict['gamma'], gpr])
            rows.append((loss_dict['wmse'], loss_dict['gamma'], gpr))

            pred_np = output_norm[0, 0].float().cpu().numpy()
            true_np = target[0, 0].float().cpu().numpy()
            mid = pred_np.shape[1] // 2

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(true_np[:, mid, :], cmap='jet', vmin=0, vmax=1)
            axes[0].set_title("Target")
            axes[1].imshow(pred_np[:, mid, :], cmap='jet', vmin=0, vmax=1)
            axes[1].set_title("Predicted")
            axes[2].imshow(pred_np[:, mid, :] - true_np[:, mid, :], cmap='bwr', vmin=-0.3, vmax=0.3)
            axes[2].set_title("Difference")
            for ax in axes:
                ax.set_xticks([]); ax.set_yticks([])
            fig.suptitle(f"Patient {index_list_test[i]} | GPR {gpr:.2f}% | WMSE {loss_dict['wmse']:.4f}")
            fig.tight_layout()
            fig.savefig(os.path.join(image_folder, f"patient_{index_list_test[i]}.png"), dpi=120)
            plt.close(fig)

            print(f"[{i+1}/{test_examples}] patient {index_list_test[i]}: WMSE={loss_dict['wmse']:.4f}, GPR={gpr:.2f}%")

        arr = np.array(rows)
        means = arr.mean(axis=0)
        stds = arr.std(axis=0)
        writer.writerow([])
        writer.writerow(['MEAN', means[0], means[1], means[2]])
        writer.writerow(['STD', stds[0], stds[1], stds[2]])

    print("\n" + "=" * 50)
    print(f"Test complete on {len(rows)} patients")
    print(f"Mean WMSE: {means[0]:.6f} | Mean Gamma: {means[1]:.4f} | Mean GPR: {means[2]:.2f}% (+/-{stds[2]:.2f})")
    print(f"Results CSV: {results_csv}")
    print(f"Images: {image_folder}")


if DATA_MODE == 'tile':
    run_tile_mode()
else:
    run_volume_mode()
