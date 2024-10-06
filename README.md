# Planetary Seismic Event Detection

## Project Overview

This project aims to solve a critical challenge in planetary seismology: reducing the power and bandwidth required to transmit continuous seismic data from space missions back to Earth. By distinguishing seismic signals from noise, only the relevant seismic events are sent, optimizing the use of power and bandwidth.

We leverage machine learning to develop an efficient model for identifying seismic events within noisy planetary data, such as those collected by the Apollo missions and the Mars InSight Lander. The goal is to minimize the amount of irrelevant data transmitted by detecting the start of seismic events and ignoring the noise.

## Approach

Our approach combines several machine learning techniques to create a robust and accurate detection system:

1.  **CNN + RNN Architecture**:
    -   We use a Convolutional Neural Network (CNN) to capture spatial patterns in the seismic waveform data.
    -   A Recurrent Neural Network (RNN) is used to model the sequential nature of seismic events, enhancing the temporal understanding of the signal.

2.  **Vision-Based Model (U Net)**:
    -   We incorporate a vision model to extract useful features from spectrograms of the seismic data, which helps to differentiate between noise and actual events.

3.  **Pre-Trained Model (Phasenet)**:
    -   We fine-tune the pre-trained **Phasenet** model, which is specifically designed for seismic event detection. Phasenet is already well-suited for recognizing seismic phases and will complement the other models.

4.  **Ensemble Voting Mechanism**:
    - We run all the above model(and maybe models with other architectures) in an ensemble system.
    - The final decision on whether the received signal is noise or an actual seismic event is made through a **voting system**, where the majority vote determines the outcome.

## How It Works
1.  The seismic data is passed to the CNN + RNN model, U-Net model, and a fine-tuned Phasenet model.
2.  Each model processes the data independently and outputs a decision: whether the input is noise or a valid seismic event.
3.  The decisions are then combined using a majority voting system. If the majority of the models detect a seismic event, the data is flagged as relevant; otherwise, it is classified as noise.

This ensures robustness, as the ensemble of models increases the accuracy of detecting seismic events within noisy data.

## Data Sources
-   **Apollo Missions**: Seismic data recorded by the Apollo lunar missions, particularly focusing on moonquakes. 
    - Source: https://atmos.nmsu.edu/data_and_services/atmospheres_data/INSIGHT/insight.html

-   **Mars InSight Lander**: Seismic data collected from Mars, including marsquakes and other planetary phenomena.
    -   Source: https://atmos.nmsu.edu/data_and_services/atmospheres_data/INSIGHT/insight.html

- We used the dataset provided by the NASA SpaceApps challenge
  - Source: https://wufs.wustl.edu/SpaceApps/data/space_apps_2024_seismic_detection.zip

## Getting Started

### Prerequisites

-   Python 3
-   PyTorch
-   NumPy
-   Pandas
-   Matplotlib
-   ObsPy (for reading seismic data)
-   SciPy
-   Pre-trained Phasenet model and U-Net model

### Installation

1.  Clone this repository:   
    
    `https://github.com/Dawn-Of-Justice/hacking-nasa-with-html.git` 
    
2.  Install the required packages:
    
    `pip install -r requirements.txt` 
    
## License
This project is licensed under the MIT License. See the LICENSE file for details.
