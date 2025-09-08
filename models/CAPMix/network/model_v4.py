import torch
import torch.nn as nn
import torch.nn.functional as F


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()

        self.input_projection = Conv1d(configs.input_channels, configs.project_channels, 1)
        self.output_projection = Conv1d(configs.project_channels, 1, 1)

        self.conv_block1 = nn.Sequential(
            Conv1d(configs.project_channels, configs.project_channels, 3, padding=1, dilation=1),
            nn.BatchNorm1d(configs.project_channels),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            # nn.Dropout(self.dropout)
        )

        self.conv_block2 = nn.Sequential(
            Conv1d(configs.project_channels, configs.project_channels, 3, padding=2, dilation=2),
            nn.BatchNorm1d(configs.project_channels),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            Conv1d(configs.project_channels, configs.project_channels, 3, padding=4, dilation=4),
            nn.BatchNorm1d(configs.project_channels),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        # self.projection_head = nn.Sequential(
        #     nn.Linear(self.final_out_channels * self.features_len, self.final_out_channels * self.features_len // self.project),
        #     nn.BatchNorm1d(self.final_out_channels * self.features_len // self.project),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.final_out_channels * self.features_len // self.project, 2),
        # )
        self.logits = nn.Linear(configs.window_size, 2)

    def forward(self, x_in):
        if torch.isnan(x_in).any():
            print('tensor contain nan')

        x = self.input_projection(x_in)
        x = F.relu(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.output_projection(x)
        x = F.relu(x)
        print("output_projection shape:", x.shape)

        x = x.reshape(x.size(0), -1)
        logits = self.logits(x)

        return logits
