import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def load_model(model_path, device):
    model = CNNModel().to(device)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Epoch: {epoch}, Loss: {loss}")
        return model
    else:
        raise FileNotFoundError(f"No model found at {model_path}")

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    logger.info(f"Preprocessing data from {file_path}")
    if 'filename' in df.columns:
        df = df.drop('filename', axis=1)
    if 'arrival_time' in df.columns:
        true_times = df['arrival_time'].values
        df = df.drop('arrival_time', axis=1)
    else:
        true_times = None
    
    df.fillna(df.mean(), inplace=True)
    array = df.to_numpy()
    X = array.reshape((-1, 2555, 79))
    return X, true_times

def run_inference(model, data, device):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(data).float().to(device)
        outputs = model(inputs)
        # Multiply by 10000 to convert back to original scale
        predictions = outputs.cpu().numpy() * 10000
    return predictions

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load the model
    model = load_model(args.model_path, device)
    
    try:
        # Load and preprocess data
        X, true_times = preprocess_data(args.data_path)
        logger.info(f"Data loaded and preprocessed. Shape: {X.shape}")
        
        # Run inference
        predictions = run_inference(model, X, device)
        
        # Save results
        results_dir = 'inference_results'
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = os.path.join(results_dir, f'predictions_{timestamp}.csv')
        
        results_df = pd.DataFrame({
            'Predicted_Time': predictions.flatten()
        })
        
        if true_times is not None:
            results_df['True_Time'] = true_times
            results_df['Difference'] = results_df['Predicted_Time'] - results_df['True_Time']
            mse = ((results_df['Predicted_Time'] - results_df['True_Time']) ** 2).mean()
            logger.info(f"Mean Squared Error: {mse}")
        
        results_df.to_csv(results_path, index=False)
        logger.info(f"Results saved to {results_path}")
        
        # Print some sample predictions
        logger.info("\nSample Predictions:")
        logger.info(results_df.head())
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seismic Model Inference")
    parser.add_argument("--model_path", type=str, help="Path to the model file")
    parser.add_argument("--data_path", type=str, default='processed_seismic_data.csv', help="Path to the input data CSV file")
    args = parser.parse_args()

    if args.model_path is None:
        # Find the most recent model file
        model_files = glob.glob('models/seismic_model_*_final.pth')
        if not model_files:
            raise FileNotFoundError("No model files found in the models directory")
        args.model_path = max(model_files, key=os.path.getctime)
        logger.info(f"Using most recent model: {args.model_path}")

    main(args)