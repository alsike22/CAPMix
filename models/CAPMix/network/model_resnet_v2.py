import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class base_Model(nn.Module):
    def __init__(self, params):
        super(base_Model, self).__init__()
        self.params = params
        self.input_projection = Conv1d(params.input_channels, params.residual_channels, 4)
        self.residual_layers = nn.ModuleList([
            ResidualBlock(params.residual_channels, 2 ** (i % params.dilation_cycle_length))
            for i in range(params.residual_layers)
        ])
        self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 4)
        self.output_projection = Conv1d(params.residual_channels, 1, 4)
        self.seq_len = params.window_size - 9
        nn.init.zeros_(self.output_projection.weight)

        self.projection_head = nn.Sequential(
            nn.Linear(self.seq_len, self.seq_len // 2),
            nn.BatchNorm1d(self.seq_len // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.seq_len // 2, 2),
        )
        # self.logits = nn.Linear(self.final_out_channels * self.features_len, 2)



    def forward(self, x_in):
        if torch.isnan(x_in).any():
            print('tensor contain nan')
        # print("x shape:", x_in.shape)
        x = self.input_projection(x_in)
        x = F.relu(x)

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x)
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        # print("x2 shape:", x.shape)
        x = self.output_projection(x)
        # print("output shape:", x.shape)
        hidden = x.reshape(x.size(0), -1)
        logits = self.projection_head(hidden)
        return logits

class ResidualBlock(nn.Module):
  def __init__(self, residual_channels, dilation):
    '''
    :param n_mels: inplanes of conv1x1 for spectrogram conditional
    :param residual_channels: audio conv
    :param dilation: audio conv dilation
    :param uncond: disable spectrogram conditional
    '''
    super().__init__()
    # padding= dilation*(kernel_size−1)/2 保持序列长度不变
    # 如果 kernel_size 是奇数，可以直接设置 padding = (kernel_size - 1) // 2
    self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 5, padding=2 * dilation, dilation=dilation)
    print("dilation shape:", dilation)
    self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

  def forward(self, x):
    y = x
    y = self.dilated_conv(y)
    gate, filter = torch.chunk(y, 2, dim=1)
    y = torch.sigmoid(gate) * torch.tanh(filter)
    # print("y1 shape:", y.shape)
    y = self.output_projection(y)
    # print("y2  shape:", y.shape)
    residual, skip = torch.chunk(y, 2, dim=1)
    # print("residual shape:", residual.shape)
    # print("skip shape:", skip.shape)
    # print("x1 shape:", x.shape)
    return (x + residual) / sqrt(2.0), skip