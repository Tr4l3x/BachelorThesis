import torch
import inference
import project.network.simple_UNet
import matplotlib.pyplot as plt

# Determine the used device for whole process
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the same type of network architecture
model = project.network.simple_UNet.SimpleUNet()
model.to(device)

# Determine the path to the network used for inference
evaluation_path = 'SimpleUNetResult.pth'

# Loading the model state at the determined path
model.load_state_dict(torch.load(evaluation_path))


num_of_plots = 10

for i in range(num_of_plots):
    result = inference.sample__plot_image(model, device=device)
    result = result.to("cpu")

    plt.imshow(result[0, 0, :, :], origin='lower')

    plt.title(str(i)+'. Generierter Takt')
    plt.xlabel('Ticks')
    plt.ylabel('Pitch')

    plt.savefig('plots\generate_bar '+str(i))
