"""
Test a trained checkpoint on the Test set: per-patient WMSE / gamma pass rate,
a summary CSV, and target-vs-predicted-vs-difference images.
"""
import os
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network_2_1 import unet
from data_pipeline_256_32_256 import data_pipeline
from losses_opt_1mm import GammaIndexLoss, WeightedMSE, CombinedWMSEGammaLoss

np.random.seed(42)
torch.manual_seed(42)

# ===================== CONFIG: cambia questi prima di ogni test =====================
MODEL_PATH = "/media/proton-lab/EXTERNAL_USB/matteo_thesis/models_and_outputs/2026-09-21_21-11-32_tile_gamma_combined_wmse_gamma_train_2_1_out_256-32-256_zero_outside_more_data_2mm_1%.pth"
USE_TILING = 'tile'          # deve combaciare con come e' stato allenato il modello
TILE_SHAPE = (128, 16, 16)   # deve combaciare con la TILE_SHAPE usata in fase di training per QUESTO checkpoint
config_tag = "tile_gamma_v2"    # solo per nominare gli output
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


test_examples = len([d for d in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, d))])
index_list_test = np.array([str(i) for i in range(test_examples)])
print(f"Found {test_examples} test patients")

base_test_dataset = data_pipeline(
    path=test_path,
    index_list=index_list_test,
    threshold=0.1,
    mode=DATA_MODE,
    tile_shape=TILE_SHAPE,
)
test_dataset = NoAugmentTestSet(base_test_dataset, test_examples)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

# ===================== MODEL =====================
model = unet(input_shape=input_shape)
model = model.to(device)

X_sample, _ = next(iter(test_dataloader))
input_channels = X_sample.shape[1]
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

# ===================== TEST LOOP =====================
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

        # ---- Save comparison image (central slice along the short axis) ----
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

    # ---- Summary row ----
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
