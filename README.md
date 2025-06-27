ESA-GNN: Explainable Social Attention Network using Graph Attention Neural Network used for Pedestrian Trajectory Prediction

Model.py:
This project implements a pedestrian trajectory prediction model using an LSTM-based encoder-decoder combined with a Graph Attention Network (GAT) to model social interactions. The encoder captures each pedestrian's motion history, while the GAT learns inter-agent influence through attention weights. The decoder then predicts future trajectories based on both individual motion and social context.The model supports attention visualization, and outputs predicted coordinates in world or pixel space depending on the dataset setup.  

It also implements uncertainty estimation for attention based predictions.


Initially trained on benchmark datasets (ETH/UCY) using homography matrices to convert pixel ↔ world coordinates.

Later extended and validated on a custom drone video dataset, where pedestrian locations were extracted using YOLOv8 + DeepSORT and mapped to GPS + meter-based world coordinates using drone metadata (altitude, latitude, FOV).



Dataset.py (for custom NITK dataset) and Dataset_eth_ucy.py(for ETH/UCY dataset) :
The PedestrianDataset class is a custom PyTorch dataset designed to preprocess pedestrian trajectory data for training and evaluating graph-based trajectory prediction models. It reads a CSV file containing frame-wise pedestrian positions, segments the data into observation and prediction sequences, filters valid pedestrians with complete data, and encodes motion histories using an LSTM. It then constructs interaction graphs based on spatial proximity and returns graph-structured PyTorch Geometric Data objects with node features combining last observed positions and LSTM embeddings. The dataset is cached to speed up future loading.


The train.py(for custom dataset) and train_ETH_UCY.py script handles training, evaluation, and saving of a pedestrian trajectory prediction model using a graph-based neural network architecture. It loads a dataset of pedestrian trajectories, splits it into training and testing sets, and constructs a model consisting of an LSTM encoder, Graph Attention Network (GAT), and LSTM decoder. During training, the model minimizes a composite loss that includes trajectory prediction error, social interaction regularization using a social force loss, and spatial constraint penalties using a map-based loss via homography(for ETH/UCY) and via gps(for custom datsaset). The script also computes ADE/FDE metrics for evaluation, saves model checkpoints, and provides trajectory visualizations and uncertainty estimates after training.


extract_frame.py was used to get manual binary maps by drawing on the extracted frame.


utils.py
It contains essential utility functions for the ESA-GNN framework, supporting graph creation, coordinate transformation, spatial regularization, and evaluation. It includes build_graph to form interaction graphs based on pedestrian proximity and social_force_loss to discourage unrealistic closeness between agents. Several functions convert world coordinates to pixel space using homography or GPS metadata, depending on whether the dataset is ETH/UCY or NITK drone-based. Map-based loss functions like map_penalty_loss ensure predictions avoid non-walkable areas by referencing semantic maps. The module also includes compute_ade_fde for trajectory accuracy evaluation, helping the model generate socially and spatially aware predictions.

load_model.py loads model saved in saved_models folder after training to visualize predictions, uncertainty and attention for certain samples.

The visualize_prediction.py (visualize_prediction_ETH_UCY.py for ETH/UCY datasets) script performs trajectory prediction for pedestrians using a trained LSTM-GAT model and visualizes the results by overlaying the predicted, observed, and ground truth trajectories on a map image. It processes a specific data sample, performs inference using the encoder, GAT, and decoder, converts the world coordinates to pixel coordinates using a homography matrix or using gps for custom dataset, and displays the mapped trajectories using Matplotlib. This enables intuitive evaluation of the model's performance and accuracy in predicting future pedestrian paths in real-world scenes.


calibration_metrics.py
It implements calibration evaluation for uncertainty-aware trajectory predictions. The core function, evaluate_calibration, measures how well the predicted uncertainty matches the actual errors by computing empirical coverage at multiple confidence thresholds (e.g., 0.5σ, 1.0σ). It checks whether the ground-truth positions fall within predicted confidence intervals and outputs the percentage of points correctly captured at each level. This helps assess whether the model’s predicted standard deviations are reliable and meaningful for safety-critical applications like autonomous navigation.

visualize_uncertainty.py
This module visualizes uncertainty-aware trajectory predictions using Monte Carlo Dropout sampling. It runs multiple stochastic forward passes to generate a distribution of predicted future trajectories for pedestrians. The script plots observed, ground truth, mean predicted paths, and multiple sampled trajectories to show uncertainty spread. It also computes standard deviations of predictions and evaluates uncertainty calibration using metrics such as Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and Brier Score. Additionally, it provides numerical and visual outputs (like reliability diagrams) to assess how well predicted confidence intervals align with actual future positions.

The attention_visualizer.py (attention_visualizer_ETH_UCY.py for ETH/UCY dataset) script visualizes the attention weights learned by the Graph Attention Network (GAT) during pedestrian trajectory prediction by overlaying them on a corresponding video frame. It loads a specific data sample from the dataset, performs a forward pass through the encoder and GAT to extract attention weights between pedestrian nodes, converts the pedestrians' world coordinates to pixel coordinates using a homography matrix, and draws a directed graph where edge colors represent attention strengths. This annotated graph is overlaid on the actual video frame corresponding to the sample, enabling intuitive understanding of inter-pedestrian influence modeled by the GAT.








