import torch
import cv2
from model import GAT, EncoderLSTM, DecoderLSTM
from dataset import PedestrianDataset
from visualize_prediction import predict_and_visualize
from visualize_uncertainty import visualize_uncertainty
from torch.utils.data import Subset
from attention_visualizer import visualize_attention_on_video

# Load checkpoint (assuming it contains state_dicts and metadata)
checkpoint = torch.load("saved_models/model_world_coordinates_epoch100_ade0.1241_fde0.2067.pt")

map_path = "manual_mask.png"
map_image = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)

# map_shape is just the image shape
map_shape = map_image.shape

# Re-create model architectures
encoder = EncoderLSTM()
gat = GAT(in_channels=34)
decoder = DecoderLSTM(pred_len=12)

# Load the weights
encoder.load_state_dict(checkpoint["encoder_state_dict"])
gat.load_state_dict(checkpoint["gat_state_dict"])
decoder.load_state_dict(checkpoint["decoder_state_dict"])

print(f"✅ Loaded model")


# Load dataset again (same way you did during training)
dataset = PedestrianDataset("pedestrian_detector/world_coordinates/world_coordinates.csv")

predict_and_visualize(gat, encoder, decoder, dataset, sample_index=239)

# Visualize prediction + uncertainty
visualize_uncertainty(gat, encoder, decoder, dataset, sample_index=239, T=100)


visualize_attention_on_video(
    gat=gat,
    encoder=encoder,
    dataset=dataset,
    sample_index=239,  # 🔁 Change to your desired pedestrian sample
    video_path="pedestrian_detector/Canteen_Dense.mp4",
)

