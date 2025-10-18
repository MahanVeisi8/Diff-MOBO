import sys,os
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../.."))
sys.path.append(os.path.abspath("../../.."))
from src.diffusion_core.diffusion import GaussianDiffusion1D
from src.diffusion_core.model import Unet1D
import torch
import os,sys
# Add the parent directory of Gen_src to sys.path
import torch
import os
import matplotlib.pyplot as  plt

# Same architecture as in training
model = Unet1D(
    dim=32,
    dim_mults=(2, 4, 8, 16),
    channels=2,  # X and Y
    dropout=0.1
).cuda()  # or .to(device)

# Create the same diffusion wrapper
diffusion = GaussianDiffusion1D(
    model,
    seq_length=192,      # must match your training setup
    objective='pred_noise',
    timesteps=1000,
    auto_normalize=True
).cuda()  # or .to(device)

# Load checkpoint
checkpoint_path = rf"/home/bardiya/projects/diffusion_air_manifolding/codes/creative-generativeai-diffusion/src/diffusion_notebooks/DIffusion_model_weigths_and_datas/dpp_0.1_autonorm_true_125_from_base_ddpm/model_epoch_124.pt"
model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
model.eval()
print("Loaded model weights from:", checkpoint_path)


# B, C, N = 1, 2, 192
B ,  C ,N = 1,2,192
latent = torch.randn((B, C, N), device="cuda")

# Save exactly (requested) t = 500, 250, 1 (they'll be mapped if your DDIM schedule skips them)
final_img, snaps = diffusion.ddim_sample_with_snapshots(
    latent,
    timesteps_to_save=(500, 250, 1),
    clip_denoised=True,
    ddim_sigma=0.0
)

print({k: snaps[k]['actual_t'] for k in snaps})  # see which schedule t each maps to

idx = 0  # which sample in the batch to show

def plot_airfoil_tensor(sample_tensor, title=None, kind="line"):
    xy = sample_tensor.detach().cpu().numpy()  # (2, N)
    x, y = xy[0], xy[1]
    if kind == "line":
        plt.plot(x, y, linewidth=1)
    else:
        plt.scatter(x, y, s=6)
    # if title: plt.title(title)
    plt.xticks([]); plt.yticks([])
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.25, 0.25)
    plt.gca().set_axis_off()

# one figure per requested timestep for the noisy state x_t
for t in [1, 250, 500]:
    plt.figure(figsize=(4, 3))
    plot_airfoil_tensor(snaps[t]['x_t'][idx], title=f"x_t at requested t={t} (actual {snaps[t]['actual_t']})", kind="scatter")
    plt.show(); plt.close()

# one figure per requested timestep for the model's predicted clean x0 at that step
for t in [1, 250, 500]:
    plt.figure(figsize=(4, 3))
    plot_airfoil_tensor(snaps[t]['x0_pred'][idx], title=f"pred x0 at requested t={t} (actual {snaps[t]['actual_t']})", kind="line")
    plt.show(); plt.close()

# final fully denoised x0 (use final_img directly; shape (B,2,N))
plt.figure(figsize=(4, 3))
plot_airfoil_tensor(final_img[idx], title="final x0", kind="line")
plt.show(); plt.close()
