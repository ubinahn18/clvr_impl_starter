import torch
import torch.nn as nn
import math
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 32),  # First hidden layer (32 units)
            nn.ReLU(),                 # Activation function
            nn.Linear(32, 32),         # Second hidden layer (32 units)
            nn.ReLU(),                 # Activation function
            nn.Linear(32, output_size) # Output layer
        )
    
    def forward(self, x):
        return self.model(x)


class CNNEncoder(nn.Module):
    def __init__(self, input_channels, img_size, initial_channels=4):
        super(CNNEncoder, self).__init__()
        self.layers = nn.ModuleList()
        self.current_channels = initial_channels
        self.add_conv_layers(input_channels, img_size)
    
    def add_conv_layers(self, input_channels, img_size):
        in_channels = input_channels
        
        while img_size > 1:
            self.layers.append(
                nn.Conv2d(in_channels, self.current_channels, kernel_size=4, stride=2, padding=1)
            )
            self.layers.append(nn.ReLU())
            
            in_channels = self.current_channels
            self.current_channels *= 2
            img_size = img_size // 2  # Halve the size
            
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def get_image_channels_and_size(image):
    transform = transforms.Compose([
        transforms.ToTensor()  # Converts the image to a tensor with channels first (C x H x W)
    ])
    
    # Apply the transformation
    image_tensor = transform(image)
    
    # Get the channels, height, and width
    channels, height, width = image_tensor.shape
    
    return channels, height, width
    


def get_feature_dim(img_size):
    num_layers = int(math.log2(img_size))
    feature_dim = 2 * (2**num_layers)
    return feature_dim


class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, future_steps):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, input_size)  
        self.future_steps = future_steps
    
    def forward(self, x, h0, c0):
        # Unroll for input sequence
        outputs = []
        seq_length = x.size(1)
        for t in range(seq_length):
            out, (h0, c0) = self.lstm(x[:, t:t+1, :], (h0, c0))
            outputs.append(torch.tensor(h0).squeeze())

        last_output = outputs[-1]
        for _ in range(self.future_steps):
            next_input = self.fc_out(last_output)
            out, (h0, c0) = self.lstm(next_input, (h0, c0))
            last_output = self.fc_out(torch.tensor(h0).squeeze())
            outputs.append(torch.tensor(h0).squeeze())

        outputs = outputs[:-(len(outputs-seq_length)]
        outputs = torch.cat(outputs, dim=1)
        return outputs

        

class RewardPredModel(nn.Module):
    def __init__(self, reward_types, input_channels=1, img_size=64, input_steps=3, output_steps=20):
        super(RewardPredModel, self).__init__()
        self.reward_type_num = len(reward_types)
        self.encoder = CNNEncoder(input_channels=1, img_size=img_size, initial_channels=4) 
        self.feature_dim = get_feature_dim(img_size)
        self.MLP = MLP(self.feature_dim, 64)        
        self.LSTMPredictor = LSTMPredictor(64, hidden_size=10, num_layers=1, future_steps=20)
        self.RewardHeads = nn.ModuleList([
            MLP(10, 1) for _ in range(self.reward_type_num)
        ])
        self.input_steps = input_steps
        self.output_steps = output_steps

    def forward(self, img_seq):

        pred_inputs = []
        
        # Encode each image in the sequence for all batches
        for i in range(self.input_steps):
            img_batch = img_seq[:, i, :, :].unsqueeze(1)  # (batch_size, input_steps, H, W)
            print('a', img_batch.shape)
            encoded_img = self.encoder(img_batch).squeeze()  # (batch_size, feature_dim)
            print('b', encoded_img.shape)
            input_feat = self.MLP(encoded_img)  # (batch_size, feature_dim)
            print('c', input_feat.shape)
            pred_inputs.append(input_feat)
        
        pred_inputs = torch.stack(pred_inputs, dim=1)  # (batch_size, input_steps, feature_dim)
        print('d', pred_inputs.shape)

        h0 = torch.zeros(1, 32, 10)
        c0 = torch.zeros(1, 32, 10)

        pred_outputs = self.LSTMPredictor(pred_inputs, h0, c0)  # (batch_size, future_steps, hidden_size)
        print('yo' , pred_outputs.shape)
        rew_outputs = []
        for i in range(self.reward_type_num):
            rew_outputs.append(self.RewardHeads[i](pred_outputs))  # (batch_size, 1)

        rew_outputs = torch.cat(rew_outputs, dim=1).squeeze()  # (batch_size, output_steps)
        
        return rew_outputs
