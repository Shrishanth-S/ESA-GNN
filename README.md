ESA-GNN: Explainable Social Attention Network using Graph Attention Neural Network used for Pedestrian Trajectory Prediction

Model.py:
This project implements a pedestrian trajectory prediction model using an LSTM-based encoder-decoder combined with a Graph Attention Network (GAT) to model social interactions. The encoder captures each pedestrian's motion history, while the GAT learns inter-agent influence through attention weights. The decoder then predicts future trajectories based on both individual motion and social context.The model supports attention visualization, and outputs predicted coordinates in world or pixel space depending on the dataset setup.  

It also implements uncertainty estimation for attention based predictions.


Initially trained on benchmark datasets (ETH/UCY) using homography matrices to convert pixel ↔ world coordinates.

Later extended and validated on a custom drone video dataset, where pedestrian locations were extracted using YOLOv8 + DeepSORT and mapped to GPS + meter-based world coordinates using drone metadata (altitude, latitude, FOV).



Dataset.py (for custom NITK dataset) and Dataset_eth_ucy.py(for ETH/UCY dataset) :
The PedestrianDataset class is a custom PyTorch dataset designed to preprocess pedestrian trajectory data for training and evaluating graph-based trajectory prediction models. It reads a CSV file containing frame-wise pedestrian positions, segments the data into observation and prediction sequences, filters valid pedestrians with complete data, and encodes motion histories using an LSTM. It then constructs interaction graphs based on spatial proximity and returns graph-structured PyTorch Geometric Data objects with node features combining last observed positions and LSTM embeddings. The dataset is cached to speed up future loading.

