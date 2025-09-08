import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    # nn.init.kaiming_normal_(layer.weight)
    return layer

# 去掉了残差结构
class base_Model(nn.Module):
    def __init__(self, params):
        super(base_Model, self).__init__()
        self.params = params
        self.input_projection = Conv1d(params.input_channels, params.residual_channels, 4)
        self.conv_layers = nn.ModuleList([
            ConvBlock(params.residual_channels, 2 ** (i % params.dilation_cycle_length))
            for i in range(params.residual_layers)
        ])
        self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 4)
        self.output_projection = Conv1d(params.residual_channels, 1, 4)
        self.seq_len = params.window_size - 6
        # nn.init.zeros_(self.output_projection.weight)

        self.projection_head = nn.Sequential(
            nn.Linear(self.seq_len, self.seq_len // 2),
            nn.BatchNorm1d(self.seq_len // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.seq_len // 2, 2),
        )
        self.logits = nn.Linear(self.seq_len, 2)



    def forward(self, x_in):
        if torch.isnan(x_in).any():
            print('tensor contain nan')
        # print("x shape:", x_in.shape)
        x = self.input_projection(x_in)
        x = F.relu(x)

        for layer in self.conv_layers:
            x = layer(x)
        # x = self.skip_projection(x)
        # print("x2 shape:", x.shape)
        x = self.output_projection(x)
        x = F.relu(x)
        # print("output shape:", x.shape)
        hidden = x.reshape(x.size(0), -1)
        logits = self.logits(hidden)
        return logits

class ConvBlock(nn.Module):
  def __init__(self, residual_channels, dilation):
    '''
    :param n_mels: inplanes of conv1x1 for spectrogram conditional
    :param residual_channels: audio conv
    :param dilation: audio conv dilation
    :param uncond: disable spectrogram conditional
    '''
    super().__init__()
    self.dilated_conv = Conv1d(residual_channels, residual_channels, 5, padding=2 * dilation, dilation=dilation)
    # print("dilation shape:", dilation)
    # self.output_projection = Conv1d(2 * residual_channels, residual_channels, 1)

  def forward(self, x):
    # y = x
    y = self.dilated_conv(x)
    # y = self.output_projection(y)
    # print("y2  shape:", y.shape)
    # print("residual shape:", residual.shape)
    # print("skip shape:", skip.shape)
    # print("x1 shape:", x.shape)
    # print("y1 shape:", y.shape)
    return y
