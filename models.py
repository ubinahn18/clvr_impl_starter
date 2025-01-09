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
    feature_dim = img_size // (2**num_layers)
    return feature_dim


class PredictorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, future_steps, batch_first=True):
        super(PredictorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        outputs = []
        h, (hidden, cell) = self.lstm(x)  # Encode input sequence
    
        return h


class RewardPredModel(nn.Module):
    def __init__(self, input_channels=3, img_size=64, input_steps=3, output_steps=20):
        super(RewardPredModel, self).__init__()
        self.encoders = nn.ModuleList([
            CNNEncoder(input_channels=3, img_size=img_size, initial_channels=4) 
            for _ in range(input_steps)
        ])
        self.feature_dim = get_feature_dim(img_size)
        self.MLP = MLP(self.feature_dim, 64)        
        self.PredictorLSTM = PredictorLSTM(64, hidden_size=20, num_layers=1, future_steps=20, batch_first=True)
        self.RewardHeads = nn.ModuleList([
            MLP(20, 1) for _ in range(output_steps)
        ])

    def forward(self, img_seq):
        pred_inputs = []
        rew_outputs = []
        i = 0
        for img in img_seq:
            encoded_img = self.encoders[i](img)
            input_feat = self.MLP(encoded_img)
            pred_inputs.append(input_feat)
            i += 1
        pred_inputs = torch.stack(pred_inputs, dim=0)
        pred_outputs = self.PredictorLSTM.forward(pred_inputs)
        for i in range(output_steps):
            rew_outputs.append(self.RewardHeads[i](pred_outputs))
        return torch.tensor(rew_outputs)









