import math
import torch
import torch.nn as nn

"""
   For later time embeddings
"""

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim-1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

"""
   Block implementation for the UNet 
"""
class Block(nn.Module):

    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, kernel_size=3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.bnorm = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t, ):
        # First Conv.
        h = self.bnorm(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extended last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channels
        h = h + time_emb
        # Second Conv.
        h = self.bnorm(self.relu(self.conv2(h)))
        # En- or decode
        return self.transform(h)

"""
   First implementation of a simple UNet structure 
"""

class SimpleUNet(nn.Module):

    def __init__(self):
        super().__init__()

        # TODO: Input channels 32??? Pro Sample aber nur eine Eingabe!!!
        input_channels = 1
        encoder_channels = (64, 128, 256)
        decoder_channels = (256, 128, 64)
        output_channels = 1
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Input
        self.input = nn.Conv2d(input_channels, encoder_channels[0], kernel_size=3, padding=1)

        # Encoder Part
        self.encoder = nn.ModuleList([Block(encoder_channels[i], encoder_channels[i+1],
                                            time_emb_dim)
                                      for i in range(len(encoder_channels)-1)])
        # Decoder Part
        self.decoder = nn.ModuleList([Block(decoder_channels[i], decoder_channels[i+1],
                                            time_emb_dim, up=True)
                                      for i in range(len(decoder_channels) - 1)])

        # Output
        self.output = nn.Conv2d(decoder_channels[-1], 1, output_channels)

    def forward(self, x, timestep):
        # Embedded time
        t = self.time_mlp(timestep)
        # Input convolution
        x = self.input(x)
        # For later concatenating when decoding
        residual_inputs = []
        # Encoding pass
        for block in self.encoder:
            x = block(x, t)
            residual_inputs.append(x)
        # Decoder pass
        for block in self.decoder:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = block(x, t)
        return self.output(x)

model = SimpleUNet()
print("Num params: ", sum(p.numel() for p in model.parameters()))


