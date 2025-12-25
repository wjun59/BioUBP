import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F


class VDCNN(nn.Module):
    def __init__(self, input_size, output_size, num_embedding=16):
        super(VDCNN, self).__init__()

        self.embedding = nn.Embedding(input_size, num_embedding)

        self.conv_block1 = self._conv_block(num_embedding, 64)
        self.conv_block2 = self._conv_block(64, 128)
        self.conv_block3 = self._conv_block(128, 256)
        self.conv_block4 = self._conv_block(256, 512)

        self.fc = nn.Linear(512, output_size)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.embedding(x.long())
        x = x.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        x = F.adaptive_max_pool1d(x, 1).squeeze(2)
        output = self.fc(x)
        return output



