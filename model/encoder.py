
from torch import nn


class TFencoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=8,
                      stride=4, bias=False, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, configs.projection, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.projection),

            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.Dropout1 = nn.Dropout(p=configs.dropout)



    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x=self.Dropout1(x)
        return x

