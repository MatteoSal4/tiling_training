import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from network_2_1 import unet
from data_pipeline_256_32_256 import data_pipeline
import numpy as np
import csv
import os
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import datetime
from losses_opt_1mm import GammaIndexLoss, SimpleBetaScheduler, WeightedMSE, CombinedWMSEGammaLoss
import wandb
import matplotlib.pyplot as plt
import pymedphys

## JUST TRAINING, NO RETRAING ##

# WITH NORMALIZATION, ZERO OUTSIDE WEIGHT, MORE DATA##

DOSE_PERCENT_THRESHOLD = 2.0
DTA_MM_THRESHOLD = 2.0
DOSE_CUTOFF = 0.2
VOXEL_SIZE_MM = 2.0 # Voxel is isometric with 2mm sides, so we can use the same value for all dimensions


scaler = GradScaler()
input_shape = (256, 32, 256)
input_shape_str = "256-32-256"
train_batch_size = 4
validation_batch_size = 4
epochs = 150

# ===================== CONFIG: cambia questi due prima di ogni run =====================
USE_TILING = 'tile'    # None (volume intero) | 'tile' | 'rec'
USE_GAMMA = False    # True -> gamma pesa nella loss | False -> solo WMSE ('basic')
TILE_SHAPE = (64, 16, 64)

DATA_MODE = USE_TILING if USE_TILING else 'none'
config_tag = f"{DATA_MODE}_{'gamma' if USE_GAMMA else 'basic'}"
# ==========================================================================================

train_path = '/media/proton-lab/migrameter_/data/Train/'
validation_path = '/media/proton-lab/migrameter_/data/Val/'
test_path = '/media/proton-lab/migrameter_/data/Test/'

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_csv = f"/media/proton-lab/migrameter_/models_and_outputs/{timestamp}_{config_tag}_combined_wmse_gamma_train_2_1_out_{input_shape_str}_zero_outside_more_data_2mm_1%.csv"
save_best_model = f"/media/proton-lab/migrameter_/models_and_outputs/{timestamp}_{config_tag}_combined_wmse_gamma_train_2_1_out_{input_shape_str}_zero_outside_more_data_2mm_1%.pth"



# ========== W&B INIT ==========
wandb.init(
    project="proton-therapy-unet_2mm_voxel",
    name=f"Combined_WMSE_Gamma_loss_zero_outside_train_2_1_{timestamp}_{config_tag}_{input_shape_str}-dslow",
    config={
        "epochs": epochs,
        "train_batch_size": train_batch_size,
        "val_batch_size": validation_batch_size,
        "learning_rate": 1e-5,
        "lr_scheduler": "ReduceLROnPlateau",
        "optimizer": "Adam",
        "loss": "CombinedWMSEGammaLoss",
        "use_tiling": USE_TILING,
        "use_gamma": USE_GAMMA,
        "wmse_alpha": 1.0,
        "wmse_weight": 1.0,
        "gamma_weight": 1.0 if USE_GAMMA else 0.0,
        "outside_weight": 0.0, # was set to 0.5 in previous runs, but set to 0.0 for this one to focus on WMSE and Gamma components without outside penalty
        "outside_threshold": 0.0, # was set to 0.05 in previous runs, but set to 0.005 for this one to minimize outside penalty influence
        "dose_percent": DOSE_PERCENT_THRESHOLD,
        "dta_mm": DTA_MM_THRESHOLD,
        "voxel_size_mm": (2.0, 2.0, 2.0),
        "dose_cutoff": 0.2,
        "beta_start": 0.1,
        "beta_mid": 2.0,
        "beta_end": 5.0,
        "input_shape": input_shape,
    }
)
# ==============================

################## Train data loader ##################

training_examples = 743
index_list_train = [str(i) for i in range(0, training_examples)]
index_list_train = np.array(index_list_train)

training_generator = data_pipeline(
    path=train_path,
    index_list=index_list_train,
    threshold=0.1,
    mode=DATA_MODE,
    tile_shape=TILE_SHAPE
)
train_dataloader = DataLoader(
    training_generator,
    batch_size=train_batch_size,
    num_workers=4,
    shuffle=True,
    prefetch_factor=4,
    pin_memory=True
)

################## Validation data loader ##################

validation_examples = 250
index_list_validation = [str(i) for i in range(0, validation_examples)]
index_list_validation = np.array(index_list_validation)

validation_generator = data_pipeline(
    path=validation_path,
    index_list=index_list_validation,
    threshold=0.1,
    mode=DATA_MODE,
    tile_shape=TILE_SHAPE
)
val_dataloader = DataLoader(
    validation_generator,
    batch_size=validation_batch_size,
    num_workers=4,
    shuffle=False,
    prefetch_factor=4,
    pin_memory=True
)

################################### Training Setup ###################################

device = "cuda"

# ===================== MODEL LOADING =====================
model = unet(input_shape=input_shape)
model = model.to(device)

# Get input channels from data
X_sample, _ = next(iter(train_dataloader))
input_channels = X_sample.shape[1]
print(f"Input channels: {input_channels}")

# Initialize lazy layers with dummy forward pass
print("Initializing model...")
dummy = torch.randn(1, input_channels, *input_shape, device=device)
with torch.no_grad():
    _ = model(dummy)
del dummy
print(f"✔ Model initialized with {len(model.state_dict())} keys")

# ===================== LOSS FUNCTIONS =====================

# WMSE Loss
wmse_loss = WeightedMSE(alpha=1.0).to(device)

# Gamma Index Loss
gamma_loss = GammaIndexLoss(
    dose_percent=DOSE_PERCENT_THRESHOLD,
    dta_mm=DTA_MM_THRESHOLD,
    voxel_size_mm=(VOXEL_SIZE_MM, VOXEL_SIZE_MM, VOXEL_SIZE_MM),
    dose_cutoff=0.2,
    beta_init=0.1,
    max_gamma=10.0,
).to(device)

# Combined Loss
criterion = CombinedWMSEGammaLoss(
    wmse_loss=wmse_loss,
    gamma_loss=gamma_loss,
    wmse_weight=1.0,
    gamma_weight=1.0 if USE_GAMMA else 0.0,
    outside_weight=0.0, # was set to 0.5 in previous runs, but set to 0.0 for this one to focus on WMSE and Gamma components without outside penalty
    outside_threshold=0.0, # was set to 0.05 in previous runs, but set to 0.005 for this one to minimize outside penalty influence
    momentum=0.99,
).to(device)

# ===================== BETA SCHEDULER =====================
beta_scheduler = SimpleBetaScheduler(
    loss_fn=gamma_loss,  # Pass gamma_loss directly, not criterion
    beta_start=0.1,
    beta_mid=2.0,
    beta_end=5.0,
    warmup_iters=10,
    phase1_end=50,
    phase2_end=100,
    update_interval=5,
)

# ===================== OPTIMIZER & SCHEDULER =====================
optimizer = optim.Adam(model.parameters(), lr=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.7, patience=10, min_lr=1e-6
)

# W&B watch model
wandb.watch(model, log="gradients", log_freq=100)

best_val_gpr = 0.0

################################### Training Loop ###################################

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    train_gpr = 0.0
    train_wmse = 0.0
    train_gamma = 0.0
    train_outside = 0.0

    current_beta = beta_scheduler.step(epoch)

    for X, Y in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        input = X.to(device, non_blocking=True)
        target = Y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(dtype=torch.bfloat16):
            output = model(x=input)

            output_norm = output / (output.amax(dim=(2, 3, 4), keepdim=True) + 1e-10)
            # the target is already normalized to [0, 1], so we can directly compute the loss without additional normalization

            loss, loss_dict = criterion(output_norm, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = X.size(0)
        train_loss += loss.item() * batch_size
        train_wmse += loss_dict['wmse'] * batch_size
        train_gamma += loss_dict['gamma'] * batch_size
        train_outside += loss_dict['outside'] * batch_size

        with torch.no_grad():
            gpr = criterion.compute_pass_rate(output_norm, target)
            train_gpr += gpr * batch_size

        # Free memory
        del input, target, output, output_norm, loss
    torch.cuda.empty_cache()

    n_train = len(train_dataloader.dataset)
    avg_train_loss = train_loss / n_train
    avg_train_gpr = train_gpr / n_train
    avg_train_wmse = train_wmse / n_train
    avg_train_gamma = train_gamma / n_train
    avg_train_outside = train_outside / n_train

    # ===== Validation =====
    model.eval()
    val_loss = 0.0
    val_gpr = 0.0
    val_wmse = 0.0
    val_gamma = 0.0
    val_outside = 0.0

    with torch.no_grad():
        with autocast(dtype=torch.bfloat16):
            for X, Y in val_dataloader:
                input = X.to(device, non_blocking=True)
                target = Y.to(device, non_blocking=True)

                output = model(x=input)
                output_norm = output / (output.amax(dim=(2, 3, 4), keepdim=True) + 1e-10)
                # the target is already normalized to [0, 1], so we can directly compute the loss without additional normalization

                loss, loss_dict = criterion(output_norm, target)
                gpr = criterion.compute_pass_rate(output_norm, target)

                batch_size = input.size(0)
                val_loss += loss.item() * batch_size
                val_gpr += gpr * batch_size
                val_wmse += loss_dict['wmse'] * batch_size
                val_gamma += loss_dict['gamma'] * batch_size
                val_outside += loss_dict['outside'] * batch_size

                del input, target, output, loss, output_norm
    torch.cuda.empty_cache()


    n_val = len(val_dataloader.dataset)
    avg_val_loss = val_loss / n_val
    avg_val_gpr = val_gpr / n_val
    avg_val_wmse = val_wmse / n_val
    avg_val_gamma = val_gamma / n_val
    avg_val_outside = val_outside / n_val

    print(f"\nEpoch {epoch+1}/{epochs} | Beta: {current_beta:.4f}")
    print(f"  Train - Loss: {avg_train_loss:.6f}, GPR: {avg_train_gpr:.2f}%")
    print(f"          WMSE: {avg_train_wmse:.6f}, Gamma: {avg_train_gamma:.4f}, Outside: {avg_train_outside:.6f}")
    print(f"  Val   - Loss: {avg_val_loss:.6f}, GPR: {avg_val_gpr:.2f}%")
    print(f"          WMSE: {avg_val_wmse:.6f}, Gamma: {avg_val_gamma:.4f}, Outside: {avg_val_outside:.6f}")

    # ========== W&B LOGGING ==========
    wandb.log({
        "epoch": epoch + 1,
        "beta": current_beta,
        "train/loss": avg_train_loss,
        "train/gpr": avg_train_gpr,
        "train/wmse": avg_train_wmse,
        "train/gamma": avg_train_gamma,
        "train/outside": avg_train_outside,
        "val/loss": avg_val_loss,
        "val/gpr": avg_val_gpr,
        "val/wmse": avg_val_wmse,
        "val/gamma": avg_val_gamma,
        "val/outside": avg_val_outside,
        "lr": optimizer.param_groups[0]['lr'],
    })
    # ==================================

    # Log to CSV
    with open(log_csv, mode='a', newline='') as f:
        writer = csv.writer(f)
        if epoch == 0:
            writer.writerow([
                'epoch', 'beta',
                'train_loss', 'train_gpr', 'train_wmse', 'train_gamma', 'train_outside',
                'val_loss', 'val_gpr', 'val_wmse', 'val_gamma', 'val_outside'
            ])
        writer.writerow([
            epoch + 1, current_beta,
            avg_train_loss, avg_train_gpr, avg_train_wmse, avg_train_gamma, avg_train_outside,
            avg_val_loss, avg_val_gpr, avg_val_wmse, avg_val_gamma, avg_val_outside
        ])

    # Save best model based on GPR
    if avg_val_gpr > best_val_gpr:
        best_val_gpr = avg_val_gpr
        torch.save(model.state_dict(), save_best_model)
        print(f"  ✔ Saved best model with GPR {best_val_gpr:.2f}%")
        print(f"    {save_best_model}")

        # W&B artifact
        artifact = wandb.Artifact(
            name=f"unet-combined-wmse-gamma-{config_tag}-{input_shape_str}",
            type="model",
            metadata={
                "val_gpr": avg_val_gpr,
                "epoch": epoch + 1,
                "beta": current_beta,
                "val_wmse": avg_val_wmse,
                "val_gamma": avg_val_gamma,
            }
        )
        artifact.add_file(save_best_model)
        wandb.log_artifact(artifact)

    scheduler.step(avg_val_gpr)
    torch.cuda.empty_cache()

print("\n" + "="*50)
print("Training complete!")
print(f"Best validation GPR: {best_val_gpr:.2f}%")
print("="*50)


################################### TESTING ###################################

print("\n" + "="*60)
print("Starting Testing with Best Model...")
print("="*60)

# Output folders for test results
prediction_folder = f"/NFSHOME/mspezialetti/sharedFolder/3D_Unet/new_experiments/predictions_combined_wmse_gamma_{timestamp}"
image_folder = os.path.join(prediction_folder, "images")
os.makedirs(prediction_folder, exist_ok=True)
os.makedirs(image_folder, exist_ok=True)

# Load best model
print(f"Loading best model from: {save_best_model}")
state_dict = torch.load(save_best_model, map_location=device)
model.load_state_dict(state_dict)
model.eval()
del state_dict
torch.cuda.empty_cache()
print("✔ Best model loaded")
