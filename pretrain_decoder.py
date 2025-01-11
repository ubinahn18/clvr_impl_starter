import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torch.optim as optim
from models import *
from sprites_env.envs.sprites import *


class Conv2D_params:

    def __init__(
        self,
        dimn_tensor=[None, None, None, None],
        hidden_layers_list=None,
        ksize=None,
        latent_space_dimn=None,
    ):

        self.batchsize = dimn_tensor[0]
        self.channels = dimn_tensor[1]
        self.nX = dimn_tensor[2]
        self.nY = dimn_tensor[3]
        self.hidden_layers_list = hidden_layers_list
        self.ksize = ksize
        self.latent_space_dimn = latent_space_dimn


class Decoder_2D(nn.Module):
    
    def __init__(self, latent_dimn, fc_outputsize, nX, nY, channels_list, ksize):

        # Input tensors are ( batchsize , latent_dimn )

        super(Decoder_2D, self).__init__()

        n_layers = len(channels_list) - 1
        ksize = ksize[::-1]

        self.f_linear_in = nn.Linear(latent_dimn, fc_outputsize)

        nn.init.xavier_uniform_(self.f_linear_in.weight)

        self.f_conv = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    channels_list[i],
                    channels_list[i + 1],
                    kernel_size=ksize[i],
                    padding=(ksize[i] - 1) // 2,
                )
                for i in range(n_layers)
            ]
        )

        for conv_i in self.f_conv:
            nn.init.xavier_uniform_(conv_i.weight)

        self.fc_outputsize = fc_outputsize

        self.channels_list = channels_list

        self.nX = nX
        self.nY = nY
        self.nx_conv = nX
        self.ny_conv = nY

    def forward(self, x):

        x = self.f_linear_in(x).reshape(
            x.size()[0], self.channels_list[0], self.nx_conv, self.ny_conv
        )

        for i, conv_i in enumerate(self.f_conv[:-1]):
            x = conv_i(x)
            x = F.relu(x)

        x = self.f_conv[-1](x)

        return x


def get_decoder2d_fcoutputsize_from_encoder2d_params(
    encoder_hidden_layers_list, ksize, nX, nY
):
    """ Calculate parameters for constructing decoder"""

    decoder_channels = encoder_hidden_layers_list[-1::-1]

    # n_layers = len( encoder_hidden_layers_list ) - 1

    len_signal_conv_X = nX
    len_signal_conv_Y = nY

    fc_outputsize = len_signal_conv_X * len_signal_conv_Y * decoder_channels[0]

    return fc_outputsize


class AutoEncoder_2D(nn.Module):

    def __init__(self, dimn_tensor, hidden_layers_list, ksize, latent_space_dimn):

        super(AutoEncoder_2D, self).__init__()

        self.encoder = CNNEncoder(input_channels=1, img_size, initial_channels=4) 

        fc_outputsize = get_decoder2d_fcoutputsize_from_encoder2d_params(
            self.encoder.conv2d_params.hidden_layers_list,
            self.encoder.conv2d_params.ksize,
            self.encoder.conv2d_params.nX,
            self.encoder.conv2d_params.nY,
        )

        self.decoder = Decoder_2D(
            self.encoder.conv2d_params.latent_space_dimn,
            fc_outputsize,
            dimn_tensor[2],
            dimn_tensor[3],
            self.encoder.conv2d_params.hidden_layers_list[-1::-1],
            self.encoder.conv2d_params.ksize,
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def get_latent_space_coordinates(self, x):
        return self.encoder(x)



def train_Encoder_Decoder(env, img_size = 64, num_trajectories = 5, num_steps = 40, epochs = 20, batch_size = 32, learning_rate = 0.005):

    criterion = nn.MSELoss()
    
    losses = []

    encoder = AutoEncoder_2D.encoder(img_size = img_size)
    decoder = AutoEncoder_2D.forward()

    optimizer_encoder = optim.SGD(encoder.parameters(), lr=learning_rate)
    optimizer_decoder = optim.SGD(decoder.parameters(), lr=learning_rate)
 
    for epoch in range(epochs):
        epoch_loss = 0
        encoder.train()
        decoder.train()

        samples = []
        targets = []

        for _ in range(num_trajectories):
            _, imgs = rollout_trajectory(env, num_steps)
            i = 0
            while i + input_steps + future_steps <= len(states):
                img_seq = torch.tensor(imgs[i:i + input_steps], dtype=torch.float32) 
                targets.append(img_seq.unsqueeze(0)) 
                processed_img_seq = encoder.forward(img_seq)
                processed_img_seq = decoder.forward(processed_img_seq)
                samples.append(processed_img_seq)
                i += 1

        samples = torch.cat(samples, dim=0) 
        targets = torch.cat(targets, dim=0) 
        total_samples = samples.shape[0]
        perm = torch.randperm(total_samples)
        samples = samples[perm]
        targets = targets[perm]

        for batch_start in range(0, total_samples - total_samples % batch_size, batch_size):
            batch_end = batch_start + batch_size
            batch_samples = samples[batch_start:batch_end, :]
            batch_targets = targets[batch_start:batch_end, :]

            loss = criterion(batch_samples, batch_targets)

            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            loss.backward()
            optimizer_encoder.step()
            optimizer_decoder.step()

            epoch_loss += loss.item() * (batch_end - batch_start)

        avg_loss = epoch_loss / total_samples
        losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    torch.save(encoder.state_dict(), "encoders.pth")
    torch.save(decoder.state_dict(), "decoders.pth")

    return losses


if __name__ == "__main__":
    env = SpritesStateImgEnv()
    train_Encoder_Decoder(env, num_trajectories=50, num_steps=100, epochs=10, batch_size=32, learning_rate=0.01)
    
    

    
