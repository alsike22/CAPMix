from torch import nn
import torch
import numpy as np


def mixup_data(x, y, alpha, device):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()
        self.input_channels = configs.input_channels
        self.final_out_channels = configs.final_out_channels
        self.features_len = configs.features_len
        self.window_size = configs.window_size
        self.kernel_size = configs.kernel_size
        self.stride = configs.stride
        self.dropout = configs.dropout
        self.alpha = configs.alpha
        self.layer_mix = configs.layer_mix

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(self.input_channels, 32, kernel_size=self.kernel_size,
                      stride=self.stride, bias=False, padding=(self.kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(self.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, self.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(self.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.projection_head = nn.Sequential(
            nn.Linear(self.final_out_channels * self.features_len, self.final_out_channels * self.features_len // 2),
            nn.BatchNorm1d(self.final_out_channels * self.features_len // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.final_out_channels * self.features_len // 2, 2),
        )
        self.logits = nn.Linear(self.final_out_channels * self.features_len, 2)

    def forward(self, x_in, target, mixup_hidden = False, device='cuda'):
        if torch.isnan(x_in).any():
            print('tensor contain nan')
        # 1D CNN feature extraction
        y_a = target
        y_b = target
        lam = 1.
        if mixup_hidden == True:
            x = self.conv_block1(x_in)
            if self.layer_mix == 1:
                x, y_a, y_b, lam = mixup_data(x, target, self.alpha, device)
            x = self.conv_block2(x)
            if self.layer_mix == 2:
                x, y_a, y_b, lam = mixup_data(x, target, self.alpha, device)
            x = self.conv_block3(x)
            if self.layer_mix == 3:
                x, y_a, y_b, lam = mixup_data(x, target, self.alpha, device)
            hidden = x.permute(0, 2, 1)
            hidden = hidden.reshape(hidden.size(0), -1)
            logits = self.projection_head(hidden)
            if self.layer_mix == 4:
                logits, y_a, y_b, lam = mixup_data(logits, target, self.alpha, device)
        else:
            x = self.conv_block1(x_in)
            x = self.conv_block2(x)
            x = self.conv_block3(x)
            hidden = x.permute(0, 2, 1)
            hidden = hidden.reshape(hidden.size(0), -1)
            logits = self.projection_head(hidden)
        # print('mixup_hidden:', mixup_hidden)
        return (logits, y_a, y_b, lam) if mixup_hidden else logits
