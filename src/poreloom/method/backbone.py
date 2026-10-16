"""PoreLoom CNN backbone: the ReadCurrent convolutional architecture, vendored so PoreLoom has no runtime dependency on
third_party/. Only BasicConvResBlock and ReadCurrent are kept, unchanged (module and parameter names, initialization),
so checkpoints are interchangeable with the upstream class.

Source: Ming-Ni-Group/ReadCurrent, models/ReadCurrent.py at commit 6db8921 (third_party/ReadCurrent).

MIT License

Copyright (c) 2024 Kechen Fan

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from torch import nn


class BasicConvResBlock(nn.Module):
    def __init__(self, input_dim, n_filters, kernel_size=3, padding=1, stride=1, shortcut=False, downsample=None):
        super(BasicConvResBlock, self).__init__()
        self.downsample = downsample
        self.shortcut = shortcut

        self.conv1 = nn.Conv1d(input_dim, n_filters, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn2 = nn.BatchNorm1d(n_filters)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.shortcut:
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual

        out = self.relu(out)
        return out


class ReadCurrent(nn.Module):
    def __init__(self, n_conv_neurons, n_fc_neurons=1024, depth=9, n_classes=2, shortcut=False):
        super(ReadCurrent, self).__init__()

        layers = []
        fc_layers = []

        layers.append(nn.Conv1d(1, n_conv_neurons[0], kernel_size=19, padding=5, stride=3))
        layers.append(nn.BatchNorm1d(n_conv_neurons[0]))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool1d(2, padding=1, stride=2))
        layers.append(nn.Conv1d(n_conv_neurons[0], n_conv_neurons[1], kernel_size=3, padding=1))

        if depth == 9:
            n_conv_block_1, n_conv_block_2, n_conv_block_3, n_conv_block_4 = 1, 1, 1, 1
        elif depth == 17:
            n_conv_block_1, n_conv_block_2, n_conv_block_3, n_conv_block_4 = 2, 2, 2, 2
        elif depth == 29:
            n_conv_block_1, n_conv_block_2, n_conv_block_3, n_conv_block_4 = 5, 5, 2, 2
        elif depth == 49:
            n_conv_block_1, n_conv_block_2, n_conv_block_3, n_conv_block_4 = 8, 8, 5, 3

        layers.append(BasicConvResBlock(input_dim=n_conv_neurons[1], n_filters=n_conv_neurons[1], kernel_size=3, padding=1, shortcut=shortcut))
        for _ in range(n_conv_block_1-1):
            layers.append(BasicConvResBlock(input_dim=n_conv_neurons[1], n_filters=n_conv_neurons[1], kernel_size=3, padding=1, shortcut=shortcut))
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))  # l = initial length / 2

        ds = nn.Sequential(nn.Conv1d(n_conv_neurons[1], n_conv_neurons[2], kernel_size=1, stride=1, bias=False), nn.BatchNorm1d(n_conv_neurons[2]))
        layers.append(BasicConvResBlock(input_dim=n_conv_neurons[1], n_filters=n_conv_neurons[2], kernel_size=3, padding=1, shortcut=shortcut, downsample=ds))
        for _ in range(n_conv_block_2-1):
            layers.append(BasicConvResBlock(input_dim=n_conv_neurons[2], n_filters=n_conv_neurons[2], kernel_size=3, padding=1, shortcut=shortcut))
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))  # l = initial length / 4

        ds = nn.Sequential(nn.Conv1d(n_conv_neurons[2], n_conv_neurons[3], kernel_size=1, stride=1, bias=False), nn.BatchNorm1d(n_conv_neurons[3]))
        layers.append(BasicConvResBlock(input_dim=n_conv_neurons[2], n_filters=n_conv_neurons[3], kernel_size=3, padding=1, shortcut=shortcut, downsample=ds))
        for _ in range(n_conv_block_3 - 1):
            layers.append(BasicConvResBlock(input_dim=n_conv_neurons[3], n_filters=n_conv_neurons[3], kernel_size=3, padding=1, shortcut=shortcut))
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))  # l = initial length / 8

        ds = nn.Sequential(nn.Conv1d(n_conv_neurons[3], n_conv_neurons[4], kernel_size=1, stride=1, bias=False), nn.BatchNorm1d(n_conv_neurons[4]))
        layers.append(BasicConvResBlock(input_dim=n_conv_neurons[3], n_filters=n_conv_neurons[4], kernel_size=3, padding=1, shortcut=shortcut, downsample=ds))
        for _ in range(n_conv_block_4 - 1):
            layers.append(BasicConvResBlock(input_dim=n_conv_neurons[4], n_filters=n_conv_neurons[4], kernel_size=3, padding=1, shortcut=shortcut))

        layers.append(nn.AdaptiveMaxPool1d(8))
        fc_layers.extend([nn.Linear(8*n_conv_neurons[4], n_fc_neurons), nn.ReLU()])
        fc_layers.extend([nn.Linear(n_fc_neurons, n_fc_neurons), nn.ReLU()])
        fc_layers.extend([nn.Linear(n_fc_neurons, n_classes)])

        self.layers = nn.Sequential(*layers)
        self.fc_layers = nn.Sequential(*fc_layers)

        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.layers(x)
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)
        return out


def window_classifier():
    """The PoreLoom window classifier used for stage 0: depth 29, shortcut, 2048-unit FC head, two classes."""
    return ReadCurrent([32, 64, 128, 256, 512], n_fc_neurons=2048, depth=29, shortcut=True)
