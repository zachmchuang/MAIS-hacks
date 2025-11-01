import torch
import torch.nn as nn


class EmotionDetector(nn.Module):
    def __init__(self, layers, channels_in, channels_out, kernel_size, num_classes=6, n_fft=2048):
        super().__init__()

        self.convs = nn.ModuleList()

        self.layers = layers
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size
        self.channels_mult = channels_out

        for i in range(layers):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(self.channels_in, self.channels_out, self.kernel_size, padding="same"),
                    nn.BatchNorm2d(channels_out),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
            )
            self.channels_in = self.channels_out
            self.channels_out = self.channels_mult * self.channels_in

        '''
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, channels_out, 3, padding="same"),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            )
        channels_out = 2 * channels_out
        self.conv2 = nn.Sequential(
            nn.Conv2d(2, channels_out, 3, padding="same"),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            )
        channels_out = 2 * channels_out
        self.conv3 = nn.Sequential(
            nn.Conv2d(4, channels_out, 3, padding="same"),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            )
        '''
        self.freq_pool = nn.AdaptiveAvgPool2d((None, 256))
        # self.lstm_input_size = 64 * 8
        # self.rnn = nn.GRU(self.lstm_input_size, 16, batch_first=True, bidirectional=True)
        # self.fc = nn.Linear(16*2, num_classes)
        self.fc1 = nn.Linear(262144, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        '''
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        '''
        for conv in self.convs:
            x = conv(x)
        # x = self.freq_pool(x)
        x = self.freq_pool(x)

        x = x.flatten(start_dim=1)
        # x = x.permute(0, 3, 1, 2).contiguous().view(batch_size, x.size(3), -1)
        # x, _ = self.rnn(x)
        # x = x.mean(dim=1)
        x = self.fc1(x)
        return x


print("Memory-optimized model ready!")