

from torch import nn


# Definição da arquitetura do modelo
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.dense_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(43264, 128),
            nn.ReLU(),
            nn.Linear(128, 6400),
            nn.ReLU(),
            nn.Linear(6400, 3)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.dense_layers(x)
        return x

