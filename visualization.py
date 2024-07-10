import numpy as np
import matplotlib.pyplot as plt
import PlaygroundForMIR.model.dataset
from forward_process import *
import dataset
import torch

#  For manual tests and visualizations
test_set = PlaygroundForMIR.model.dataset.get_dataset()
test_sample = test_set.piano_rolls[0]

# Plotting some examples
fig, axes = plt.subplots(1, 10, figsize=(15,5))
for idx in range(10):
    ax = axes[idx]
    ax.imshow(test_sample[idx], origin='lower')
    ax.set_title('BarNum: '+str(idx+1))

plt.tight_layout()
plt.show()

print(test_set.num_of_rolls)

" Visualization of the forwarding process "

T = 1000
test = dataset.get_dataset()
test_roll = test.piano_rolls[0][0]
num_images = 10
stepsize = int(T/num_images)

fig, axes = plt.subplots(1, num_images+1, figsize=(15,5))
ax = axes[0]
ax.imshow(test_roll, origin='lower')
ax.set_title('Sample x0:')
for idx in range(0, T, stepsize):
    t = torch.Tensor([idx]).type(torch.int64)
    ax = axes[int(idx/stepsize)+1]
    image, noise = forward_diffusion_sample(test_roll, t)
    ax.imshow(image, origin='lower')
    ax.set_title('Timestep: '+str(idx))

plt.tight_layout()
plt.show()


