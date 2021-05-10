import torch
import torch.nn as nn
from torchvision import models


class SeqMRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained_model = models.alexnet(pretrained=True)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256 * 2, 2)
        self.classifer = nn.Linear(256, 2)  # 原版


    def forward(self, x):
        x = torch.squeeze(x, dim=0)
        features = self.pretrained_model.features(x)
        pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        zero600 = torch.zeros(600 - features.shape[0], 256).cuda()
        pooled_features = torch.cat((zero600, pooled_features), dim=0)
        pooled_features = pooled_features.unsqueeze(0).permute(1, 0, 2)
        lstm_out, hidden = self.lstm(pooled_features)
        out = self.fc(lstm_out[:, -1, :])
        return out
