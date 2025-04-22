import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader



class StravaData(Dataset):
    def __init__(self):
        # Training set
        df_train = pd.read_csv("TrainClean.csv")
        df_valid = pd.read_csv("ValidClean.csv")
        features = ['Distance','Average Speed','Average Heart Rate','Max Speed','Max Heart Rate','Idle Time','Moving Time'
        ]

        X_train = df_train[features].to_numpy(dtype=np.float32)
        X_valid = df_valid[features].to_numpy(dtype=np.float32)
        y_train = df_train['Relative Effort'].to_numpy(dtype=np.float32)
        y_valid = df_valid['Relative Effort'].to_numpy(dtype=np.float32)

        std = np.std(X_train, axis=0)
        X_train_scaled = X_train / std
        X_valid_scaled = X_valid / std

        self.X = torch.tensor(X_train_scaled)
        self.y = torch.tensor(y_train)
        # Validation set
        self.X_valid = torch.tensor(X_valid_scaled)
        self.y_valid = torch.tensor(y_valid)

        self.len = len(self.y)



    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return self.len

    def to_numpy(self):
        return np.array(self.X), np.array(self.y)


class RelativeEffort(nn.Module):
    def __init__(self):
        # Call the constructor of the super class
        super(RelativeEffort, self).__init__()

        # Single hidden layer neural network with 5 nodes in hidden layer
        self.in_to_h1 = nn.Linear(7, 64)
        self.h1_to_h2 = nn.Linear(64, 32)
        self.h2_to_h3 = nn.Linear(32, 16)
        self.h3_to_out = nn.Linear(16, 1)


    def forward(self, x):
        x = F.relu(self.in_to_h1(x))
        x = F.relu(self.h1_to_h2(x))
        x = F.relu(self.h2_to_h3(x))
        return self.h3_to_out(x).view(-1)


def trainNN(epochs=5, batch_size=16, lr=0.001):
    # Load the Dataset
    ds = StravaData()

    # Create data loader
    data_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # Create an instance of the NN
    model = RelativeEffort()

    # Mean Square Error loss function
    mse_loss = nn.MSELoss(reduction='sum')

    # Use Adam Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    running_loss = 0.0
    for epoch in range(epochs):
        for _, data in enumerate(data_loader, 0):
            x, y = data

            optimizer.zero_grad()

            output = model(x)

            loss = mse_loss(output.view(-1), y)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch} of {epochs} MSE (Train): {running_loss / len(ds)}")
        running_loss = 0.0
        with torch.no_grad():
             output = model(ds.X_valid).view(-1)
        print(f"Epoch {epoch} of {epochs} MSE (Validation): {torch.mean((output - ds.y_valid) ** 2.0)}")
        print("-" * 50)
    return model


# SVR Performance on Validation set (MSE): 634.6198690427688
# and R^2 = 0.78427481498
# SVR had C=50.0, default gamma, and normalized features
trainNN(epochs=100)
