import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import RobustScaler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StableSeismicModel(nn.Module):
    def __init__(self):
        super(StableSeismicModel, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=4, stride=4),
            
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=4, stride=4),
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        self.regressor = nn.Sequential(
            nn.Linear(16 * 8 * 8, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Check for NaN values
        if torch.isnan(x).any():
            raise ValueError("NaN values detected in input")
            
        x = x.unsqueeze(1)
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x

class NaNSafetyNet:
    @staticmethod
    def check_tensor(tensor, tensor_name):
        if torch.isnan(tensor).any():
            raise ValueError(f"NaN values detected in {tensor_name}")
        if torch.isinf(tensor).any():
            raise ValueError(f"Inf values detected in {tensor_name}")

def load_and_preprocess_data():
    logger.info('Loading the dataset...')
    df = pd.read_csv('processed_seismic_data.csv')
    
    y = df['arrival_time'].values.reshape(-1, 1)
    df = df.drop(['filename', 'arrival_time'], axis=1)
    
    logger.info('Preprocessing data...')
    # Use RobustScaler for better handling of outliers
    input_scaler = RobustScaler()
    output_scaler = RobustScaler()
    
    # Handle potential infinity and NaN values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    array = input_scaler.fit_transform(df)
    y_normalized = output_scaler.fit_transform(y)
    
    X = array.reshape(-1, 2555, 79)
    
    logger.info(f'Data shapes - X: {X.shape}, y: {y_normalized.shape}')
    logger.info(f'X stats - min: {X.min()}, max: {X.max()}, mean: {X.mean()}')
    logger.info(f'y stats - min: {y_normalized.min()}, max: {y_normalized.max()}, mean: {y_normalized.mean()}')
    
    return X, y_normalized, output_scaler

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, save_interval, model_save_path, device):
    best_val_loss = float('inf')
    nan_safety = NaNSafetyNet()
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        try:
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                nan_safety.check_tensor(inputs, "inputs")
                nan_safety.check_tensor(labels, "labels")
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                nan_safety.check_tensor(outputs, "outputs")
                
                loss = criterion(outputs, labels)
                
                if torch.isnan(loss).any():
                    logger.error(f"NaN loss detected at batch {batch_idx}")
                    continue
                
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                optimizer.step()
                
                train_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    logger.info(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.6f}')
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            train_loss = train_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)
            
            logger.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(model, optimizer, epoch, val_loss, f'{model_save_path}_best.pth')
            
            if (epoch + 1) % save_interval == 0:
                save_model(model, optimizer, epoch, val_loss, f'{model_save_path}_epoch_{epoch+1}.pth')
                
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            break
    
    save_model(model, optimizer, num_epochs, val_loss, f'{model_save_path}_final.pth')

def save_model(model, optimizer, epoch, loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    logger.info(f"Model saved to {filename}")

def main():
    # Set device and random seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Using device: {device}")

    # Hyperparameters
    BATCH_SIZE = 8
    LEARNING_RATE = 0.0001  # Reduced learning rate
    NUM_EPOCHS = 50  # Reduced epochs
    SAVE_INTERVAL = 5
    VAL_SPLIT = 0.2

    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_save_path = f'models/stable_seismic_model_{timestamp}'
    os.makedirs('models', exist_ok=True)

    # Load and preprocess data
    X, y_normalized, output_scaler = load_and_preprocess_data()
    X = torch.tensor(X).float()
    y = torch.tensor(y_normalized).float()
    
    # Create dataset and split into train/val
    dataset = TensorDataset(X, y)
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Initialize model, criterion, and optimizer
    model = StableSeismicModel().to(device)
    criterion = nn.HuberLoss(delta=1.0)  # More stable than MSE
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    logger.info('Model initialized')
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Train model
    logger.info('Starting training...')
    train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, SAVE_INTERVAL, model_save_path, device)
    logger.info('Training completed')
    
    # Save the output scaler
    import joblib
    scaler_filename = f"{model_save_path}_output_scaler.save"
    joblib.dump(output_scaler, scaler_filename)
    logger.info(f"Output scaler saved to {scaler_filename}")

if __name__ == "__main__":
    main()