import project.training
import torch
import project.network.simple_UNet
import project.network.time_embedding

# Determine the used device for whole process
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Determine the used network architecture (in network)
model = project.network.simple_UNet.SimpleUNet()
model.to(device)

# Determine training_variant and hyperparameters
training = project.training.training_variant1
T = 1000
lr = 0.001
epochs = 50
result_path = "SimpleUNetResult.pth"

# Starting training variant
training(model, device, T, lr, epochs)

# Saving the model for later evaluation via inference processing
torch.save(model.state_dict(), result_path)




