import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the size after conv and pooling layers
        self.fc1 = nn.Linear(16 * 1277 * 39, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Add channel dimension
        x = x.unsqueeze(1)  # Shape becomes (n, 1, 2555, 79)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

if __name__ == "__main__":
    # Gotta get my data in here
    import pandas as pd
    import numpy as np
    print('Loading the super big dataset. Takes a long time.')
    df = pd.read_csv('processed_seismic_data.csv')
    print('Loaded successfully')
    y = df['arrival_time']
    df = df.drop('filename', axis=1)
    df = df.drop('arrival_time', axis=1)
    print('Cleaning nan')
    df.fillna(df.mean(), inplace=True)
    print('Cleaned nan')
    array = df.to_numpy()
    y = y.values.reshape((75,1))

    X = array.reshape((-1, 2555, 79))
    print(X.shape, y.shape)
    X = torch.tensor(X).float()
    y = torch.tensor(y).float()
    # print(np.isnan(X).any(), np.isinf(X).any())
    # print(np.isnan(y).any(), np.isinf(y).any())

    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize the model
    model = CNNModel()

    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    print('Loaded model')
    num_epochs = 20
    print('Started training')
    train_model(model, train_loader, criterion, optimizer, num_epochs)
    print('Training completed')

    model.eval()
    with torch.no_grad():
        test_input = torch.randn(1, 2555, 79)
        output = model(test_input)
        print(f"Test output: {output.item()}")

    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")