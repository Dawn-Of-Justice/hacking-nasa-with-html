# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset

# class CNNModel(nn.Module):
#     def __init__(self):
#         super(CNNModel, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
#         self.relu = nn.ReLU()
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
#         # Calculate the size after conv and pooling layers
#         self.fc1 = nn.Linear(16 * 1277 * 39, 10)
#         self.fc2 = nn.Linear(10, 1)

#     def forward(self, x):
#         # x shape: (batch_size, 75, 2555, 79)
#         x = self.conv1(x)  # (batch_size, 16, 2555, 79)
#         x = self.relu(x)
#         x = self.maxpool(x)  # (batch_size, 16, 1277, 39)
#         x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# def train_model(model, train_loader, criterion, optimizer, num_epochs):
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         for inputs, labels in train_loader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
        
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# if __name__ == "__main__":
#     # Generate dummy data (replace this with your actual data)
#     num_samples = 75
#     X = torch.randn(num_samples, 2555, 79)
#     y = torch.randn(num_samples, 1)  # Assuming regression task, adjust if classification

#     # Create dataset and dataloader
#     dataset = TensorDataset(X, y)
#     train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

#     # Initialize the model
#     model = CNNModel()

#     # Define loss function and optimizer
#     criterion = nn.MSELoss()  # Mean Squared Error for regression
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     # Train the model
#     num_epochs = 2
#     train_model(model, train_loader, criterion, optimizer, num_epochs)

#     # Test the model with a single sample
#     model.eval()
#     with torch.no_grad():
#         test_input = torch.randn(1, 75, 2555, 79)
#         output = model(test_input)
#         print(f"Test output: {output.item()}")

#     # Print model summary
#     print(model)
#     print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
















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
    # Generate dummy data (replace this with your actual data)
    num_samples = 100
    X = torch.randn(num_samples, 2555, 79)
    y = torch.randn(num_samples, 1)  # Assuming regression task, adjust if classification

    # Create dataset and dataloader
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize the model
    model = CNNModel()

    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 2
    train_model(model, train_loader, criterion, optimizer, num_epochs)

    # Test the model with a single sample
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(1, 2555, 79)
        output = model(test_input)
        print(f"Test output: {output.item()}")

    # Print model summary
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")