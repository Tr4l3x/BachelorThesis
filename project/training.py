import project.dataset as dataset
import project.forward_process as forward_process
import project.inference as inference
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

"""
   Loss function used in all following training variants
"""
def get_loss(model, x_0, t, device="cpu"):
    x_noisy, noise = forward_process.forward_diffusion_sample(x_0, t, device=device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)

"""
   Training variant 0 for testing 
"""
def training_variant0(model, device="cpu", T=1000, lr=0.001, epochs=2):
    dataloader = dataset.get_dataloader()
    optimizer = Adam(model.parameters(), lr=lr)
    epochs = epochs

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            t = torch.randint(0, T, (batch.shape[0],), device=device).long()

            loss = get_loss(model, batch.float(), t, device)
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0 and step == 100:
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
                inference.sample__plot_image(model, device=device)
        print("EPOCHE VORBEIIII")
        exit()

"""
   Training variant 1 - with sample dependencies
"""
def training_variant1(model, device="cpu", T=1000, lr=0.001, epochs=2):
    # Dataset object + device(cuda) using for samples
    data = dataset.get_dataset(device)

    # Holds for each sample a list with all their piano rolls (bars)
    samples = data.piano_rolls

    # Hyperparameters
    epochs = epochs
    optimizer = Adam(model.parameters(), lr=lr)
    slide_window_size = 2

    for epoch in range(epochs):
        for sample_idx, sample in enumerate(samples):
            loss_sum = 0
            for bar_num, roll in enumerate(sample, start=1):
                optimizer.zero_grad()

                prev_bars = []
                # Case if there are not enough previous bars
                if slide_window_size > bar_num:
                    for i in range(slide_window_size-bar_num):
                        prev_bars.append(torch.tensor(np.zeros((72,48))).to(device))
                # Adding existing previous bars
                window_idx = max(0, bar_num-slide_window_size)
                for window_idx in range(window_idx, bar_num):
                    prev_bars.append(sample[window_idx])

                # Concatenating all prev. (or empty) bars with the current bar_num roll
                batch = torch.cat(prev_bars, dim=1)

                # Since training_batches should have shape (batch_size, num_of_channels, sample_dim0, sample_dim1)
                # Adding channel dimension at dim=0 (1, 72, 48*window_size)
                batch = batch.unsqueeze(0)
                # Adding channel dimension at dim=1 (1, 1, 72, 48*window_size)
                batch = batch.unsqueeze(0)

                t = torch.randint(0, T, (batch.shape[0],), device=device).long()
                loss = get_loss(model, batch.float(), t, device=device)
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()

            avg_loss = loss_sum / len(sample)
            print(f"In Epoche {epoch} fuer Sample {sample_idx} | Avg. Loss: {avg_loss} ")

            #if sample_idx == 0:
            #    inference.sample__plot_image(model, device=device)

