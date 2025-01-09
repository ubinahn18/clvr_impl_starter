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

    
    def adjust_layers(self, img_size):
        """Recompute the layers dynamically if input size is different."""
        if len(self.layers) == 0 or img_size != 64:  # 64 is the default placeholder
            self.layers = nn.ModuleList()
            self.current_channels = 16
            self.add_conv_layers(self.layers[0].in_channels if self.layers else 3)


def get_feature_dim(img_size):
    num_layers = int(math.log2(img_size))
    feature_dim = img_size // (2**num_layers)
    return feature_dim


class PredictorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, future_steps):
        super(PredictorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Predict single output (e.g., reward)

    def forward(self, x):
        outputs = []
        h, (hidden, cell) = self.lstm(x)  # Encode input sequence
        
        # Use the last hidden state to predict future steps
        last_output = h[:, -1, :]  # Get last time step's output
        for _ in range(future_steps):
            prediction = self.fc(last_output)
            outputs.append(prediction)
            last_output = prediction  # Feed prediction back as input
    
        return torch.cat(outputs, dim=1)


class RewardPredModel(nn.Module):
    def __init__(self, input_channels = 3, img_size, input_steps, output_steps):
        super(RewardPredModel, self).__init__()
        self.encoders = []
        for i in range(input_steps):
            self.encoders.append(CNNEncoder(input_channels = 3, img_size, initial_channels=4))
        feature_dim = get_feature_dim(img_size)
        self.MLP = MLP(feature_dim,64)        
        self.PredictorLSTM = PredictorLSTM(input_size, hidden_size = 20, num_layers = 1, future_steps = 20, batch_first=True)
        self.RewardHeads = []
        for i in range(future_steps):
            self.RewardHeads.append(MLP(hidden_size = 20, 1))

    def forward(self, img_seq):
        rewards = np.zeros(output_steps)
        pred_inputs = []
        for i in range(len(img_seq)):
            img = img_seq[i]
            encoded_img = self.encoders[i].forward(img)
            input_feat = self.MLP.forward(encoded_img)
            pred_inputs.append(input_feat)
        pred_inputs = np.array(pred_inputs)
        pred_outputs = self.PredictorLSTM.forward(
        
        return torch.cat(outputs, dim=1)






