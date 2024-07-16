import network
import dataset
import forward_process
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
   Loss function
"""
def get_loss(model, x_0, t):
    x_noisy, noise = forward_process.forward_diffusion_sample(x_0, t, device=device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)

"""
   Init. of the whole process 
"""

# Define beta schedule
T = 1000
betas = forward_process.linear_beta_schedule(timesteps=T)

# Pre-calculating all alphas for alpha_t=(1-beta_t)
alphas = 1. - betas
# Cumulative product of all alphas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1,0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

"""
   Sampling (Algorithm 2.) of DDPM paper
"""
@torch.no_grad()
def sample_timestep(x, t):
    betas_t = forward_process.get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = forward_process.get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = forward_process.get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = forward_process.get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


"""
   Sampling (Algorithm 2.) of DDPM paper
"""
@torch.no_grad()
def sample__plot_image():
    # Sample noise
    img = torch.randn((1, 1, 72, 48), device=device)
    plt.figure(figsize=(5, 5))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)

    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize+1))
    plt.show()


"""
   Training 
"""
from torch.optim import Adam

dataloader = dataset.get_dataloader()
model = network.SimpleUNet().to(device)
optimizer = Adam(model.parameters(), lr=0.001)
batch_size = 32
epochs = 2

for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        t = torch.randint(0, T, (batch.shape[0],), device=device).long()
        loss = get_loss(model, batch.float(), t)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 and step == 100:
            print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
            sample__plot_image()
    print("EPOCHE VORBEIIII")
    exit()