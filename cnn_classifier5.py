
from torch import nn


class Classifier5(nn.Module):
    def __init__(self):
        super(Classifier5, self).__init__()

        #Camadas convolucionais
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=44, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=44, out_channels=128, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

        )

        # Define as camadas densas
        self.dense_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(15488, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
            )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.dense_layers(x)
        return x