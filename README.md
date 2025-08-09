# ESA-GNN: Explainable Social Attention Network for Pedestrian Trajectory Prediction

This repository contains the implementation of ESA-GNN, an Explainable Social Attention Network using a Graph Attention Neural Network for pedestrian trajectory prediction. The model combines an LSTM-based encoder-decoder architecture with a Graph Attention Network (GAT) to effectively model both individual pedestrian motion and social interactions.

## Key Features

* **Hybrid Architecture:** Employs an LSTM-based encoder to capture individual motion history and a GAT to model inter-agent influence through learned attention weights.

* **Explainability:** The GAT's attention weights can be visualized to understand which pedestrians the model considers most influential to a target pedestrian's trajectory.

* **Uncertainty Estimation:** Integrates Monte Carlo Dropout for robust uncertainty-aware predictions, essential for safety-critical applications like autonomous navigation.

* **Dataset Versatility:** Initially trained on benchmark datasets (ETH/UCY) using homography matrices, the model was later extended and validated on a custom drone video dataset (NITK), mapping pedestrian locations to GPS and meter-based world coordinates.

* **Spatial Awareness:** Includes mechanisms like map-based loss to ensure predicted trajectories avoid non-walkable areas.

## Project Structure & Core Scripts

### `Model.py`

This script defines the core pedestrian trajectory prediction model. It uses an LSTM-based encoder-decoder and a Graph Attention Network (GAT). The model supports attention visualization and can output predictions in either world or pixel coordinates. It also includes the implementation for uncertainty estimation.

### Dataset Scripts (`Dataset.py` & `Dataset_eth_ucy.py`)

The `PedestrianDataset` class handles the preprocessing of trajectory data. It reads CSV files, segments data into observation and prediction sequences, filters valid pedestrians, and constructs PyTorch Geometric `Data` objects with graph structures based on spatial proximity.

### Training Scripts (`train.py` & `train_ETH_UCY.py`)

These scripts manage the end-to-end training and evaluation process. They load the dataset, build the model, and minimize a composite loss function that includes:

* Trajectory prediction error

* A social force loss for social interaction regularization

* A map-based loss to enforce spatial constraints
  The scripts also compute standard evaluation metrics like Average Displacement Error (ADE) and Final Displacement Error (FDE), and save model checkpoints.

### Visualization & Analysis Scripts

* **`visualize_prediction.py` (`visualize_prediction_ETH_UCY.py`):** Visualizes the predicted, observed, and ground truth trajectories by overlaying them on a map image.

* **`visualize_uncertainty.py`:** Generates and visualizes uncertainty-aware predictions using Monte Carlo Dropout. It plots multiple sampled trajectories, computes standard deviations, and evaluates calibration using metrics like ECE and MCE.

* **`attention_visualizer.py` (`attention_visualizer_ETH_UCY.py`):** Overlays the GAT's attention weights onto a video frame, showing a directed graph where edge colors represent attention strengths, providing a clear visual explanation of social influence.

### Utility Scripts

* **`utils.py`:** A collection of essential functions for the framework, including `build_graph` for creating interaction graphs, `social_force_loss` for social regularization, and functions for coordinate transformation (world to pixel) and trajectory evaluation.

* **`load_model.py`:** A helper script to load a pre-trained model for visualization and analysis purposes.

* **`calibration_metrics.py`:** Contains functions to evaluate the reliability of uncertainty predictions by comparing predicted confidence intervals with actual errors.

## Getting Started

### Prerequisites

* \[List of required libraries, e.g., PyTorch, PyTorch Geometric, NumPy, Matplotlib\]

### Installation

1. Clone the repository:

   ```
   git clone [https://github.com/Shrishanth-S/ESA-GNN.git]
   cd ESA-GNN
   
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   
   ```

### Datasets

* Annotations for the custom NITK dataset are located in the `pedestrian_detector` folder.

* Annotations for the ETH/UCY dataset are located in the `data` folder.

### Running a Script

* To train the model: `python train.py` (for the custom dataset) or `python train_ETH_UCY.py` (for ETH/UCY).

* To visualize predictions: `python visualize_prediction.py` (after training).

* To visualize attention: `python attention_visualizer.py`.
