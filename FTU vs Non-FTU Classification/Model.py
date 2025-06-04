import torch
import torch.nn as nn


class FTUCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), padding=1 ), # in : (3, 224, 224)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2), # out: (16, 112, 112) 

            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1 ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2), # out: (32, 56, 56) 

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1 ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2), # out: (64, 28, 28) 

            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)

        )

    def forward(self, X):
        return self.network(X)
