import torch
import torch.nn as nn
import numpy as np

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 1277 * 39, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_model(model_path):
    model = CNNModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_input(input_data):
    # Assuming input_data is a numpy array of shape (2555, 79)
    # Convert to torch tensor and add batch dimension
    return torch.tensor(input_data).float().unsqueeze(0)

def perform_inference(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
    return output.item() * 1000.0  # Convert back to original scale

def main():
    # Load the trained model
    model_path = 'path_to_your_saved_model.pth'
    model = load_model(model_path)

    # Example: Load and preprocess your input data
    # Replace this with your actual data loading logic
    input_data = np.random.randn(2555, 79)  # Example random input
    input_tensor = preprocess_input(input_data)

    # Perform inference
    predicted_arrival_time = perform_inference(model, input_tensor)
    print(f"Predicted arrival time: {predicted_arrival_time:.4f} ms")

if __name__ == "__main__":
    main()